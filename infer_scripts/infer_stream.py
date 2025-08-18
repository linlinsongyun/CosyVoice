# -*- coding: utf-8 -*-
import sys
import os
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import time
import os
import sys
import argparse
#import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
import scipy.io.wavfile
import onnxruntime
import whisper
import math
import torchaudio.compliance.kaldi as kaldi
from cosyvoice.utils.common import fade_in_out
#from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
option = onnxruntime.SessionOptions()
option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
option.intra_op_num_threads = 1

speech_tokenizer_session1 = onnxruntime.InferenceSession("CosyVoice/pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx",
                                                        sess_options=option,
                                                        providers=[
                                                            "CUDAExecutionProvider" if torch.cuda.is_available() else
                                                            "CPUExecutionProvider"])

speech_tokenizer_session = onnxruntime.InferenceSession("CosyVoice/pretrained_models/speech_tokenizer_v2.onnx",
                                                        sess_options=option,
                                                        providers=[
                                                            "CUDAExecutionProvider" if torch.cuda.is_available() else
                                                            "CPUExecutionProvider"])

campplus_session = onnxruntime.InferenceSession("CosyVoice/pretrained_models/campplus.onnx",
                                                sess_options=option, providers=["CPUExecutionProvider"])

MAX_WAV_VALUE = 32768
rtconfig = onnxruntime.SessionOptions()
rtconfig.intra_op_num_threads = 1
    


def extract_speech_token(speech_tokenizer_session, wav, device):
    feat = whisper.log_mel_spectrogram(wav, n_mels=128)
    speech_token = speech_tokenizer_session.run(None,
                                                {speech_tokenizer_session.get_inputs()[0].name:
                                                     feat.detach().cpu().numpy(),
                                                 speech_tokenizer_session.get_inputs()[1].name:
                                                     np.array([feat.shape[2]], dtype=np.int32)})[
        0].flatten().tolist()
    speech_token = torch.tensor([speech_token], dtype=torch.int32).to(device)
    speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(device)
    print('speech_token', speech_token.device)
    return speech_token, speech_token_len


def extract_spk_embedding(speech, device):
    feat = kaldi.fbank(speech,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = campplus_session.run(None,
                                     {campplus_session.get_inputs()[0].name: feat.unsqueeze(
                                         dim=0).cpu().numpy()})[0].flatten().tolist()
    embedding = torch.tensor([embedding]).to(device)
    print('embedding', embedding.device)
    return embedding

def denorm_mel_lne2log10(mel, eps=1e-6):
    mel = np.exp(mel)
    mel = np.log10(np.maximum(eps, mel))
    return mel


def stream_infer2_bak(mel, ort_session, chunk_size=50, ovl=25, sr=24000, hop_size=240):
    # hop_size: 10ms
    ovl_samples = int(ovl * hop_size)
    chunk_samples = int(chunk_size * hop_size)
    stream_audio_chunks = []
    B, T, D = mel.shape
    #print(mel.shape, cond.shape)
    for i, start in enumerate(range(0, T - ovl, chunk_size + ovl)):
        s = time.time()
        end = min(start + ovl + chunk_size + ovl, T)
        print(start, end, T)
        #wav_tmp = model.forward_onnx(mel[:,start:end, :])[0][0]
        ort_inputs = {ort_session.get_inputs()[0].name: mel[:,start:end, :]}
        wav_tmp = ort_session.run(None, ort_inputs)[0][0][0]

        #if i == 0:
        #    latency = time.time() - s
        stream_audio_chunks.append(wav_tmp)
    audio = segs_fade_merge(stream_audio_chunks)
    return audio, latency
    audio = stream_audio_chunks[0][:chunk_samples + ovl_samples]
    index = np.linspace(0, 1, ovl_samples)
    for i in range(1, len(stream_audio_chunks)):
        #print(stream_audio_chunks[i].shape)
        tmp = (1 - index) * stream_audio_chunks[i - 1][-ovl_samples :] + index * stream_audio_chunks[i][:ovl_samples]
        t = min(ovl_samples + chunk_samples, len(stream_audio_chunks[i]))
        audio = np.hstack((audio, tmp, stream_audio_chunks[i][ovl_samples : t]))
    audio = np.hstack((audio, stream_audio_chunks[-1][t:]))
    return audio

def stream_infer2(mel_stream_list, ort_session, chunk_size=10, ovl=4, sr=24000, hop_size=240):
    # hop_size: 10ms
    ovl_samples = int(ovl * hop_size)
    #chunk_samples = int(chunk_size * hop_size)
    stream_audio_chunks = []
    #B, T, D = mel.shape
    #print(mel.shape, cond.shape)
    for mel in mel_stream_list:
        s = time.time()
        #print('mel', mel.shape)
        mel = np.expand_dims(mel, axis=0)
        ort_inputs = {ort_session.get_inputs()[0].name: mel}
        wav_tmp = ort_session.run(None, ort_inputs)[0][0][0]
        
        latency = time.time() - s
            
        stream_audio_chunks.append(wav_tmp)
        
    audio = segs_fade_merge(stream_audio_chunks, chunk_size=chunk_size, ovl=ovl)
    return audio

def stream_cosy2(mel_stream_list, ort_session, chunk_size=10, ovl=4, sr=24000, hop_size=240):
    # hop_size: 10ms
    ovl_samples = int(ovl * hop_size)
    #chunk_samples = int(chunk_size * hop_size)
    stream_audio_chunks = []
    #B, T, D = mel.shape
    #print(mel.shape, cond.shape)
    for mel in mel_stream_list:
        s = time.time()
        #print('mel', mel.shape)
        mel = np.expand_dims(mel, axis=0)
        tts_speech, tts_source = cosyvoice.model.hift.inference(mel=tts_mel)
        
        ort_inputs = {ort_session.get_inputs()[0].name: mel}
        wav_tmp = ort_session.run(None, ort_inputs)[0][0][0]
        
        latency = time.time() - s
            
        stream_audio_chunks.append(wav_tmp)
        
    audio = segs_fade_merge(stream_audio_chunks, chunk_size=chunk_size, ovl=ovl)
    return audio

    
    
def segs_fade_merge_bak(pcms, frameMinSize=1000, ovl=25, sr=24000, hop_size=240):
    ovl_samples = int(ovl * hop_size)
    chunk_samples = int((frameMinSize / 10) * hop_size)
    print('segs_fade_merge over mel', ovl)
    assert chunk_samples > ovl_samples > 1
    delta = 1 / (ovl_samples - 1)
    index_r = [delta * i for i in range(ovl_samples)]
    index_l = [delta * i for i in range(ovl_samples-1, -1, -1)]

    res = list(pcms[0][:chunk_samples + ovl_samples])
    #res.append(audio)
    print([len(t) for t in pcms])
    for i in range(1, len(pcms)):
        tmp = pcms[i-1][-ovl_samples:]
        pcm = pcms[i]
        print(len(pcm), len(tmp), ovl_samples)
        for j in range(ovl_samples):
            pcm[j] = tmp[j] * index_l[j] + pcm[j] * index_r[j]
        end = min(ovl_samples + chunk_samples, len(pcm))
        res.extend(pcm[:end])
        
    
    #print('segs_fade_merge', len(res[0]), res[-1])
    #res = np.array(res)
    return res


def stream_wav_concat(tts_speech_lists, save_wav_path, speech_chunk_len, speech_overlap_len):

    
    delta = 1 / (speech_overlap_len - 1)
    index_r = [delta * i for i in range(speech_overlap_len)]
    index_l = [delta * i for i in range(speech_overlap_len-1, -1, -1)]
    
    print('tts_mel_lists', len(tts_speech_lists))
    #speech_window = np.hamming(2 * speech_overlap_len)
    #half_window_len = int(speech_overlap_len/2)
    speech_cache = None
    audio_final = np.zeros(1)
    for audio in tts_speech_lists:

        #print('audio', audio.shape)
        if speech_cache is None:
            audio_final = np.concatenate((audio_final, audio))
        else:
            audio_overlap = speech_cache*index_l + audio[:speech_overlap_len]*index_r
            
            audio_final[-speech_overlap_len:] = audio_overlap
            #print('start', start/24000, 'end', end/24000, 'speech_overlap_len', speech_overlap_len)
            
            audio = audio[speech_overlap_len:]
            #print('audio_final', len(audio_final)/24000, 'audio', len(audio)/24000)
            audio_final = np.concatenate((audio_final, audio))
        speech_cache = audio[-speech_overlap_len:]
        '''
            audio_overlap = audio[-half_window_len:] * speech_window[:half_window_len] + speech_cache*speech_window[-half_window_len:]
            audio_final[-half_window_len:] = audio_overlap
            audio = audio[half_window_len:]
            audio_final = np.concatenate((audio_final, audio))
        speech_cache = audio[-half_window_len:]
        '''

    print('audio_final', len(audio_final)/24000, len(audio_final))
    audio = audio_final / max(audio_final) * 0.7
    audio = audio * MAX_WAV_VALUE
    audio = audio.astype('int16')
    
    scipy.io.wavfile.write(save_wav_path, 24000, audio)
    
        
        
def segs_fade_merge(pcms, chunk_size=1000, ovl=25, sr=24000, hop_size=240):
    ovl_samples = int(ovl * hop_size)
    chunk_samples = int(chunk_size * hop_size)
    #chunk_samples = int((frameMinSize / 10) * hop_size)
    #print('segs_fade_merge over mel', ovl)
    assert chunk_samples > ovl_samples > 1
    delta = 1 / (ovl_samples - 1)
    index_r = [delta * i for i in range(ovl_samples)]
    index_l = [delta * i for i in range(ovl_samples-1, -1, -1)]
    
    #print('segs_fade_merge pcm', pcms[0].size())
    res = list(pcms[0][:chunk_samples + ovl_samples])
    #res.append(audio)
    print('pcms', [len(t) for t in pcms])
    for i in range(1, len(pcms)):
        tmp = pcms[i-1][-ovl_samples:]
        pcm = pcms[i]
        print('pcm', len(pcm), 'tmp', len(tmp), 'ovl_samples', ovl_samples)
        for j in range(ovl_samples):
            pcm[j] = tmp[j] * index_l[j] + pcm[j] * index_r[j]
        
        end = min(ovl_samples + chunk_samples, len(pcm))
        res.extend(pcm[:end])
        
    
    #print('segs_fade_merge', len(res[0]), res[-1])
    #res = np.array(res)
    return res

def flow_matching_infer_stream(prompt_audio_path, tts_audio_path, device, n_timesteps, chunk_size=10, overlap_size=4, sr=16000, mel_dim=160):

        # torch.audio [-1,1]
    wav = load_wav(prompt_audio_path, sr) # [1, T]
    
    # 500ms
    
    #.squeeze(0) 
    #print('=====wav dur', wav.size(1)/sr)
    #sr, audio = prompt_audio
    #wav = torch.tensor(audio)
    #wav = wav.to(device) / 32767.0
    #if wav.ndim == 2:
    #   print("=========wav double channel into single")
    #    wav = wav.mean(-1)
    #wav = wav.unsqueeze(0)
    #print("=========wav unsqueeze", wav.size())
    gain_db = -24 - math.log10(torch.mean(wav * wav).cpu().item() + 1e-16) * 10
    wav = torchaudio.functional.gain(wav, gain_db)
    wav = torch.clip(wav, min=-1, max=1)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    speech_token, speech_token_len = extract_speech_token(speech_tokenizer_session, wav, device)
    embedding = extract_spk_embedding(wav, device)
    #speech_token = speech_token[:, :25]
    #speech_token_len = 25
    print('speech_token', speech_token.size()) #, speech_token.device)
    print('embedding', embedding.size()) #, embedding.device)
    # 
    
    start_time = time.time()
    tts_mel_stream = []
    tts_speech_stream = []
    tts_mel_len_sum = 0
    tts_source = torch.zeros(1, 1, 0).to(device)
    for i, start in enumerate(range(0, speech_token_len - overlap_size, chunk_size + overlap_size)):
   
        end = min(start + overlap_size + chunk_size, speech_token_len)
        #print(start, end)
        speech_token_chunk = speech_token[:, start:end]
        

        speech_token_chunk_len = speech_token_chunk.size(1)
        # tts_mel:[B, 160, T]
        print('start:', start, 'end:', end)
        print('speech_token_chunk_len', speech_token_chunk_len, 'token overlap_size', overlap_size)
        tts_mel, _ = cosyvoice.model.flow.inference(token=speech_token_chunk,
                                                 token_len=speech_token_chunk_len,
                                                 prompt_token=torch.zeros(1, 0, dtype=torch.int32).to(device),
                                                 prompt_token_len=torch.zeros(1, dtype=torch.int32).to(device),
                                                 prompt_feat=torch.zeros(1, 0, mel_dim).to(device),
                                                 prompt_feat_len=torch.zeros(1, dtype=torch.int32).to(device),
                                                 embedding=embedding,
                                                 n_timesteps=n_timesteps,
                                                 finalize=True)
        #print('tts_mel', tts_mel.size(), 'tts_source', tts_source.size(), tts_source.size(2)/24000)
        # tts_mel torch.Size([1, 80, 36]), tts_source 0.72
        # 最后一个片段


        if tts_source.size(2) !=0 and tts_mel.size(2) * 2* 240 !=tts_source.size(2):
            tts_source_len = tts_mel.size(2) * 2 * 240
            print('tts_mel', tts_mel.size(), 'tts_source_len', tts_source_len)
            tts_source = torch.cat((tts_source, tts_source), dim=2)
            tts_source = tts_source[:, :, -tts_source_len:]
        #print('into model tts_mel', tts_mel.size(), 'tts_source', tts_source.size(2)/24000)
        tts_speech, tts_source = cosyvoice.model.hift.inference(speech_feat=tts_mel, cache_source=tts_source)
        #print('tts_source', tts_source.size(2)/24000)
        tts_audio_part_dir = os.path.join(os.path.dirname(os.path.dirname(tts_audio_path)), 'part_wav')
        os.makedirs(tts_audio_part_dir, exist_ok=True)
        tts_audio_part_path = os.path.join(tts_audio_part_dir, os.path.basename(tts_audio_path).replace('.wav', '_%d.wav'%i))                          
        tts_speech = tts_speech.detach().cpu()                                                
        torchaudio.save(tts_audio_part_path, tts_speech, 24000)
        print('tts_speech', tts_speech.size())
        tts_speech_stream.append(tts_speech)

        
    #print('======tts_mel_len_sum', tts_mel_len_sum, 'token segs: ', len(tts_mel_stream))
    vq_rate = 25 # hz
    # 1 token = 40ms, 1 mel_frame = 20ms
    mel_chunk_size = chunk_size * 2
    mel_overlap_size = overlap_size * 2
    audio = segs_fade_merge(tts_speech_stream, chunk_size=mel_chunk_size, ovl=mel_overlap_size, sr=24000, hop_size=480)
    
    torchaudio.save(tts_audio_path, audio, 24000)
    


    return tts_audio_path

def extract_speech_token_emb(prompt_audio_path, vq_path, device, sr=16000, token_mel_ratio=1):
    #def flow_matching_infer_stream(prompt_audio_path, tts_audio_path, device, n_timesteps, chunk_size=10, overlap_size=4, sr=16000, mel_dim=160):

        # torch.audio [-1,1]
    wav = load_wav(prompt_audio_path, sr) # [1, T]
    
    # 500ms
    
    #.squeeze(0) 
    #print('=====wav dur', wav.size(1)/sr)
    #sr, audio = prompt_audio
    #wav = torch.tensor(audio)
    #wav = wav.to(device) / 32767.0
    #if wav.ndim == 2:
    #   print("=========wav double channel into single")
    #    wav = wav.mean(-1)
    #wav = wav.unsqueeze(0)
    #print("=========wav unsqueeze", wav.size())
    gain_db = -24 - math.log10(torch.mean(wav * wav).cpu().item() + 1e-16) * 10
    wav = torchaudio.functional.gain(wav, gain_db)
    wav = torch.clip(wav, min=-1, max=1)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if vq_path is not None:
        speech_token = torch.from_numpy(np.load(vq_path).reshape(1, -1)).to(device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(device)
        prompt_token, prompt_token_len = extract_speech_token(speech_tokenizer_session1, wav, device)
    else:
        speech_token, speech_token_len = extract_speech_token(speech_tokenizer_session1, wav, device)
        prompt_token, prompt_token_len = speech_token, speech_token_len 
    print('src speech_token', speech_token.size())
    embedding = extract_spk_embedding(wav, device)
    #speech_token = speech_token[:, :25]
    #speech_token_len = 25
    #last_chunk_len = speech_token_len%25 * token_mel_ratio
    #chunk_times = speech_token_len//25 + 1
    #pad_size = (0, chunk_times*25+3-speech_token_len)
    #padded_tensor = torch.nn.functional.pad(speech_token, pad=pad_size, mode='constant', value=0)
    #speech_token = padded_tensor.to(device)
    src_wav_dur = speech_token.size(1)*token_mel_ratio*480/24000.
    
    
    print('speech_token', speech_token.size()) #, speech_token.device)
    print('embedding', embedding.size()) #, embedding.device)
    print('prompt audio', wav.size())
    #print('last_chunk_len', last_chunk_len)
   # sys.exit()
    return speech_token, embedding, src_wav_dur, 0

def extract_mel(audio_path):
    from matcha.utils.audio import mel_spectrogram
    n_fft =  1920
    num_mels =  80
    sampling_rate = 24000
    hop_size =  480
    win_size =  1920
    fmin = 0
    fmax =  8000

    wav = load_wav(audio_path, sampling_rate)
    mel = mel_spectrogram(wav, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
    print('mel:', mel.size())
    return mel


def extract_token25_emb_mel(prompt_audio_path, vq_path, device, prompt_token_path=None, sr=16000, token_mel_ratio=1, with_pad=True, first_hop_len=25, sub_hop_len=0):
    # extract emb & mel from prompt_audio_path
    #print('prompt_audio_path', prompt_audio_path)
    prompt_feat = extract_mel(prompt_audio_path)
    #print('prompt_feat', prompt_feat.size())
    prompt_feat = prompt_feat.transpose(1,2).to(device)
    #print('prompt_feat', prompt_feat.size(), prompt_feat.device)
    if prompt_token_path is None:
        prompt_token_path = os.path.join(os.path.dirname(vq_path), os.path.basename(prompt_audio_path).replace('.wav', '.npy'))
    else:
        print('use given prompt_token_path:', prompt_token_path)
        
    prompt_token = torch.from_numpy(np.load(prompt_token_path)).to(device).transpose(0,1)
    #print('extract_token25_emb_mel src prompt_token', prompt_token.size())
    
    wav = load_wav(prompt_audio_path, sr) # [1, T]
    
    # 500ms
    
    #.squeeze(0) 
    #print('=====wav dur', wav.size(1)/sr)
    #sr, audio = prompt_audio
    #wav = torch.tensor(audio)
    #wav = wav.to(device) / 32767.0
    #if wav.ndim == 2:
    #   print("=========wav double channel into single")
    #    wav = wav.mean(-1)
    #wav = wav.unsqueeze(0)
    #print("=========wav unsqueeze", wav.size())
    gain_db = -24 - math.log10(torch.mean(wav * wav).cpu().item() + 1e-16) * 10
    wav = torchaudio.functional.gain(wav, gain_db)
    wav = torch.clip(wav, min=-1, max=1)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        

    speech_token = torch.from_numpy(np.load(vq_path).reshape(1, -1)).to(device)
    speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(device)

    #print('extract_token25_emb_mel src speech_token', speech_token.size())
    embedding = extract_spk_embedding(wav, device)
    #speech_token = speech_token[:, :25]
    #speech_token_len = 25

    last_chunk_len = 0
    src_wav_dur = float(speech_token_len/25.)
    #print('after pad :', with_pad, 'speech_token', speech_token.size()) 
    #, speech_token.device)
    #print('embedding', embedding.size()) #, embedding.device)
    #print('prompt audio', wav.size())
    #print('last_chunk_len', last_chunk_len)
    #sys.exit()
    return speech_token, embedding, src_wav_dur, last_chunk_len, prompt_token, prompt_feat



def pad_token(speech_token_len, token_mel_ratio, first_hop_len=25, sub_hop_len=0):
    print('first_hop_len', first_hop_len, 'sub_hop_len', sub_hop_len)
    if sub_hop_len==0:
        last_chunk_len = speech_token_len%first_hop_len * token_mel_ratio
        
        #chunk_times = speech_token_len//25 + 1
        pad_size = (0, (first_hop_len+3)*token_mel_ratio-last_chunk_len)
        #print('sub_hop_len=0', pad_size)
        #print('pad_size', pad_size)
    else:
        
        last_chunk_len = (speech_token_len-first_hop_len)%sub_hop_len * token_mel_ratio
        
        #chunk_times = speech_token_len//25 + 1
        pad_size = (0, (sub_hop_len+3)*token_mel_ratio-last_chunk_len)
    print('speech_token_len', speech_token_len, 'last_chunk_len', last_chunk_len)
    print('pad_size', pad_size)
    #sys.exit()
    return pad_size, last_chunk_len

        
def token2wav(this_speech_token, embedding, prompt_token, prompt_feat, flow_cache_dict={}, hift_cache_dict={}, finalize=False, trim_len=50, mel_dim=80, mel_cache_len=8, last_chunk=False):
    #with torch.cuda.amp.autocast(fp16):
    source_cache_len = mel_cache_len*480
    speech_window = np.hamming(2 * source_cache_len)
    speech_token_chunk_len = this_speech_token.size(1)
    # print('finalize is', finalize)
    print('token2wav this_speech_token', this_speech_token.size())
    print('token2wav speech_token_chunk_len', speech_token_chunk_len)
    # prompt_feat = torch.zeros(1, 0, mel_dim).to(device)
    # prompt_feat_len=torch.zeros(1, dtype=torch.int32).to(device),
    # prompt_token=torch.zeros(1, 0, dtype=torch.int32).to(device),
    # prompt_token_len=torch.zeros(1, dtype=torch.int32).to(device),
    try:
        
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(device)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(device)
        print('prompt_token', prompt_token.size())
        print('prompt_token_len', prompt_token_len)
        print('prompt_feat', prompt_feat.size())
        print('prompt_feat_len', prompt_feat_len)
    except:
        prompt_feat = torch.zeros(1, 0, mel_dim).to(device)
        prompt_token=torch.zeros(1, 0, dtype=torch.int32).to(device)
        prompt_feat_len = torch.zeros(1, dtype=torch.int32).to(device)
        prompt_token_len = torch.zeros(1, dtype=torch.int32).to(device)
        print(' no prompt_feat & prompt_token ')
#     print('this_speech_token', this_speech_token.size())
#     print('speech_token_chunk_len', speech_token_chunk_len)
    
#     print('embedding', embedding.size())
#     print('===========encoder_cache=========')
#     for key in flow_cache_dict['encoder_cache']:
#         try:
#             print(key, flow_cache_dict['encoder_cache'][key].size())
#         except:
#             print(key, flow_cache_dict['encoder_cache'][key])
#     print('===========decoder_cache=========')
#     for key in flow_cache_dict['decoder_cache']:
#         try:
#             print(key, flow_cache_dict['decoder_cache'][key].size())
#         except:
#             print(key, flow_cache_dict['decoder_cache'][key])
            

    # tts_mel, _ = cosyvoice.model.flow.inference(token=speech_token,
    #                                      token_len=speech_token_len,
    #                                      prompt_token=prompt_token.to(device),
    #                                      prompt_token_len=prompt_token_len.to(device),
    #                                      prompt_feat=prompt_feat.to(device),
    #                                      prompt_feat_len=prompt_token_len.to(device),
    #                                      embedding=embedding,
    #                                      streaming=False,
    #                                      finalize=True,)
        
    tts_mel, _ = cosyvoice.model.flow.inference(token=this_speech_token,
                                                 token_len=speech_token_chunk_len,
                                                 prompt_token=prompt_token, 
                                                 prompt_token_len=prompt_token_len, 
                                                 prompt_feat=prompt_feat,
                                                 prompt_feat_len=prompt_feat_len,
                                                 embedding=embedding,
                                                 streaming=False,
                                                 finalize=True)
    
    
    # mel2wav
    print('this_speech_token', this_speech_token.size(), 'tts_mel', tts_mel.size())
    np.save('test_mel.npy', tts_mel.detach().cpu().numpy())
    return tts_mel

def mel2wav_hift_nocache(cosyvoice, tts_mel, hift_cache_source=None):
    # append hift cache
    hift_cache_source = torch.zeros(1, 1, 0)
    # keep overlap mel and hift cache
    tts_speech, tts_source = cosyvoice.model.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
    
    return tts_speech

def mel2wav_hift_cache(cosyvoice, tts_mel, hift_cache_dict=None, source_cache_len=8*480, mel_cache_len=8, last_chunk=False):
    # append hift cache
    speech_window = np.hamming(2 * mel_cache_len*480)
    if hift_cache_dict is not None:
        print('hift_cache_dict is not None')
        hift_cache_mel, hift_cache_source = hift_cache_dict['mel'], hift_cache_dict['source']
        tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
    else:
        print('hift_cache_dict init')
        hift_cache_source = torch.zeros(1, 1, 0)
    # keep overlap mel and hift cache

    print('finalize is False tts_mel', tts_mel.size())
    tts_speech, tts_source = cosyvoice.model.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
    if hift_cache_dict is not None:
        tts_speech = fade_in_out(tts_speech, hift_cache_dict['speech'], speech_window)
    hift_cache_dict = {'mel': tts_mel[:, :, -mel_cache_len:],
                                  'source': tts_source[:, :, -source_cache_len:],
                                  'speech': tts_speech[:, -source_cache_len:]}
    speech_chunk_total = tts_speech
    
    if last_chunk:
        tts_speech = tts_speech
    else:
        tts_speech = tts_speech[:, :-source_cache_len]
    print('============last_chunk', last_chunk)
    print('finalize is False final tts_speech', tts_speech.size(1)/24000)
    
    return tts_speech, hift_cache_dict, speech_chunk_total

def tts_stream_generator_full(prompt_audio_path, save_wav_path, vq_path, device, prompt_token_path=None, token_hop_len=25, pre_lookahead_len=3, token_mel_ratio=1):
    # prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio=token_mel_ratio
    
    #speech_token, embedding, src_wav_dur = extract_speech_token_emb(prompt_audio_path, device)
    if token_mel_ratio==1:
        speech_token, embedding, src_wav_dur, _ = extract_speech_token_emb(prompt_audio_path, vq_path, device, token_mel_ratio=token_mel_ratio)
        prompt_feat = torch.zeros(1, 0, 80).to(device)
    elif token_mel_ratio==2:
        speech_token, embedding, src_wav_dur, _, prompt_token, prompt_feat = extract_token25_emb_mel(prompt_audio_path, vq_path, device, prompt_token_path=prompt_token_path, token_mel_ratio=token_mel_ratio)
        
    print('speech_token', speech_token, 'prompt_token', prompt_token)
    np.save('emb_happy.npy', embedding.detach().cpu().numpy())
    np.save('feat_happy.npy', prompt_feat.detach().cpu().numpy())
    
    prompt_token = prompt_token[:, -60:]
    prompt_feat = prompt_feat[:, -120:, :]
    #sys.exit()
    print('speech_token', speech_token.size())
    flow_cache_dict = None #cosyvoice.model.init_flow_cache()
    hift_cache_dict = None
    #flow_decoder_required_cache_size = self.flow.decoder.estimator.num_decoding_left_chunks * self.flow.decoder.estimator.static_chunk_size
    flow_decoder_required_cache_size = int(1* 0.5*50*1)
    # estimator.num_decoding_left_chunks = default=1/ 0.5
    # estimator.static_chunk_size = <chunk_size> * <token_frame_rate> * <token_mel_ratio>
        
    total_tokens = speech_token.size(1) #b,t, dim len(tts_speech_tokens)
    token_offset = 0
    all_speech = []
    
    tts_mel = token2wav(speech_token, embedding, prompt_token, prompt_feat,
                                 flow_cache_dict=flow_cache_dict,
                                 hift_cache_dict=hift_cache_dict,
                                 finalize=True,
                                 trim_len=flow_decoder_required_cache_size,                         
                                  )
    print('tts_mel', tts_mel.size())
    np.save('tts_mel.npy', tts_mel.detach().cpu().numpy())
    print('saved tts_mel.npy')
    speech_chunk, hift_cache_dict, speech_chunk_total = mel2wav_hift_cache(cosyvoice, tts_mel, hift_cache_dict=hift_cache_dict, last_chunk=True)
    
    #speech_chunk = speech_chunk[:, :-last_chunk_len*480]
    full_speech = speech_chunk.cpu()
    torchaudio.save(save_wav_path, full_speech, sample_rate=24000)  # 根据实际采样率调整
    print('src_wav_dur:', src_wav_dur)
    print('src_wav_dur:', src_wav_dur, 'syn_speech:', full_speech.size(1)/24000)
    print('saved ', save_wav_path)
          
    return 


# look_forward example
# git_tools/QwenTTS_0228/src/transformers/models/qwen2_code2wav_dit/modeling_qwen2_code2wav.py    
        
save_dict = {}
def tts_stream_generator(prompt_audio_path, save_wav_path, vq_path, device, prompt_token_path=None, token_hop_len=25, sub_hop_len=0, pre_lookahead_len=3, token_mel_ratio=1):
    
    if token_mel_ratio==1:
        speech_token, embedding, src_wav_dur, _ = extract_speech_token_emb(prompt_audio_path, vq_path, device, token_mel_ratio=token_mel_ratio)
        prompt_feat = torch.zeros(1, 0, 80).to(device)
        prompt_token=torch.zeros(1, 0, dtype=torch.int32).to(device)
    elif token_mel_ratio==2:
        speech_token, embedding, src_wav_dur, _, prompt_token, prompt_feat = extract_token25_emb_mel(prompt_audio_path, vq_path, device, with_pad=False, prompt_token_path=prompt_token_path, token_mel_ratio=token_mel_ratio, first_hop_len=token_hop_len, sub_hop_len=sub_hop_len)
    
    orginal_prompt_token = prompt_token
        
        


    print('speech_token', speech_token.size())
    flow_cache_dict = None #cosyvoice.model.init_flow_cache()
    hift_cache_dict = None
    #flow_decoder_required_cache_size = self.flow.decoder.estimator.num_decoding_left_chunks * self.flow.decoder.estimator.static_chunk_size
    flow_decoder_required_cache_size = int(1* 0.5*50*1)
    # estimator.num_decoding_left_chunks = default=1/ 0.5
    # estimator.static_chunk_size = <chunk_size> * <token_frame_rate> * <token_mel_ratio>
        
    total_tokens = speech_token.size(1) #b,t, dim len(tts_speech_tokens)
    token_offset = 0
    all_speech = []

    

    first_window = True
    pre_tts_mel = None
    
    token_hop_len = 50
    mel_end = token_hop_len * token_mel_ratio
    pre_lookahead_len = 25
    i = 0
    syn_dur = 0
    while True:
        
        window_length = token_hop_len + pre_lookahead_len
        available = total_tokens - token_offset

        if available >= window_length+3:
            # 正常提取
            current_tokens = speech_token[:, token_offset:token_offset + window_length+3]
            print('current_tokens st:', token_offset, 'end:', token_offset + window_length+3)
            #pad_length = 0
            print('=====================no pad branch, available:', available)
            last_chunk=False
        else:
            # 需要填充
            current_tokens = speech_token[:, token_offset:token_offset + available]
            pad_length = window_length + 3 - available
            token_pad = orginal_prompt_token[:, :pad_length]
            print('token_pad', token_pad.size(), 'current_tokens', current_tokens.size())
            current_tokens = torch.cat((current_tokens, token_pad), dim=1)
            print('current_tokens after pad', current_tokens.size())
            #, 'current_tokens', current_tokens.size())
            if pad_length >= token_hop_len or token_offset==0:
                last_chunk=True
            else:
                last_chunk=False
            print('=====================pad branch, available:', available)
            # 在 tokens 维度（dim=1）向右填充 0
            #current_tokens = F.pad(current_tokens, (0, pad_length))
            print('ava current_tokens st:', token_offset, 'end:', token_offset + available+pad_length)

        # finalize=True 不需要context
        flow_cache_dict = cosyvoice.model.init_flow_cache()
        #print('flow_cache_dict encoder_cache offset', flow_cache_dict['encoder_cache']['offset'])
        #print('flow_cache_dict decoder_cache offset', flow_cache_dict['decoder_cache']['offset'])
        print('prompt_token', prompt_token.shape, 'prompt_feat', prompt_feat.shape, 'current_tokens', current_tokens.shape)
        tts_mel = token2wav(current_tokens, embedding, prompt_token, prompt_feat,
                        flow_cache_dict=flow_cache_dict,
                        hift_cache_dict={},
                        finalize=False,
                        trim_len=flow_decoder_required_cache_size,  

                        )
        print('token2wav tts_mel', tts_mel.shape)
        # prompt_token torch.Size([1, 50]) prompt_feat torch.Size([1, 100, 80]) current_tokens torch.Size([1, 78])
        # token2wav this_speech_token torch.Size([1, 78])

#         prompt_token = None
#         prompt_feat = None
        # 缓解stream 中部分ip音量忽大忽小的问题
        tts_mel = tts_mel[:, :, :mel_end]
        prompt_feat = tts_mel.transpose(1,2)
        tts_mel_len = prompt_feat.shape[1]
        prompt_token = current_tokens[:, :tts_mel_len//token_mel_ratio]
        
        speech_chunk, hift_cache_dict, speech_chunk_total = mel2wav_hift_cache(cosyvoice, tts_mel, hift_cache_dict=hift_cache_dict, last_chunk=last_chunk)
        #speech_chunk_tmp = mel2wav_hift_nocache(cosyvoice, tts_mel_clone)
        #save_tmp_wav_path = save_wav_path.replace('.wav', '_%d.wav'%i)
        #torchaudio.save(save_tmp_wav_path, speech_chunk_tmp.detach().cpu(), sample_rate=24000)
        i = i+1
        
        cur_chunk_len = speech_chunk.size(1)/24000.
        syn_dur = syn_dur + cur_chunk_len
        print('=======cur_chunk_len', cur_chunk_len, 'total syn len:', syn_dur)


        # 更新 token_offset
        token_offset += token_hop_len
        first_window = False

        # 终止条件：当 token_offset >= total_tokens 时退出循环
        all_speech.append(speech_chunk.cpu())
        if token_offset >= total_tokens:
            break
    if len(all_speech)>1:
        last_speech_chunk = speech_chunk_total[:, :(available*token_mel_ratio+8)*480]
        last_speech_chunk_len = last_speech_chunk.size(1)/24000.
        print('last_speech_chunk_len:', last_speech_chunk_len)
        all_speech[-1] = last_speech_chunk.cpu()
    
    print('token_offset:', token_offset, 'total_tokens:', total_tokens)
    

    # 合并所有音频片段并保存
    syn_dur = 0
    for i, data in enumerate(all_speech):
        #print(data.size())
        # save_tmp_wav_path = save_wav_path.replace('.wav', '_slice_%d.wav'%i)
        # torchaudio.save(save_tmp_wav_path, data, sample_rate=24000) 

        syn_dur += data.size(1)/24000
    #save_tmp_wav_path = save_wav_path.replace('.wav', '_slice_%d.wav'%(i+1))
    #torchaudio.save(save_tmp_wav_path, all_speech[-1], sample_rate=24000) 
        
    print('syn_dur', syn_dur) 
    full_speech = torch.cat(all_speech, dim=1)
    torchaudio.save(save_wav_path, full_speech, sample_rate=24000)  # 根据实际采样率调整
    print('save_wav_path:', save_wav_path)
    print('src_wav_dur:', src_wav_dur, 'syn_speech:', full_speech.size(1)/24000)
    #sys.exit()
    save_dict[os.path.basename(save_wav_path)] = {'total_tokens':total_tokens, 'available':available, 'pad_length': pad_length, 'src_wav_dur':src_wav_dur, 'syn_speech':syn_dur}
    return 


def tts_stream_generator2(prompt_audio_path, save_wav_path, vq_path, device, prompt_token_path=None, token_hop_len=25, sub_hop_len=0, pre_lookahead_len=3, token_mel_ratio=1):
    
    if token_mel_ratio==1:
        speech_token, embedding, src_wav_dur, _ = extract_speech_token_emb(prompt_audio_path, vq_path, device, token_mel_ratio=token_mel_ratio)
        prompt_feat = torch.zeros(1, 0, 80).to(device)
        prompt_token=torch.zeros(1, 0, dtype=torch.int32).to(device)
    elif token_mel_ratio==2:
        speech_token, embedding, src_wav_dur, _, prompt_token, prompt_feat = extract_token25_emb_mel(prompt_audio_path, vq_path, device, with_pad=False, prompt_token_path=prompt_token_path, token_mel_ratio=token_mel_ratio, first_hop_len=token_hop_len, sub_hop_len=sub_hop_len)
    
    orginal_prompt_token = prompt_token
    prompt_token_ori = prompt_token
    prompt_token = prompt_token[:, :50]
    prompt_feat = prompt_feat[:, :100:, :]
    
    

    print('speech_token', speech_token.size())
    flow_cache_dict = None #cosyvoice.model.init_flow_cache()
    hift_cache_dict = None
    #flow_decoder_required_cache_size = self.flow.decoder.estimator.num_decoding_left_chunks * self.flow.decoder.estimator.static_chunk_size
    flow_decoder_required_cache_size = int(1* 0.5*50*1)
    # estimator.num_decoding_left_chunks = default=1/ 0.5
    # estimator.static_chunk_size = <chunk_size> * <token_frame_rate> * <token_mel_ratio>
        
    total_tokens = speech_token.size(1) #b,t, dim len(tts_speech_tokens)
    token_offset = 0
    all_speech = []

    

    first_window = True
    pre_tts_mel = None
    mel_cache_len = 2
    speech_window = mel_cache_len*480
    
    token_hop_len = 50
    mel_end = token_hop_len * token_mel_ratio
    pre_lookahead_len = 15
    index = 0
    syn_dur = 0
    pad_length = 0
    while True:
        
        window_length = token_hop_len + pre_lookahead_len
        available = total_tokens - token_offset
        if window_length >= total_tokens:
            current_tokens = torch.cat((speech_token, prompt_token[:, :5]), dim=1)
            last_chunk=True
        elif available >= window_length+3:
            # 正常提取
            if token_offset>0:
                token_st = token_offset-10
                token_et = token_offset + window_length+3
                print('token_offset>0 token:', token_st, token_et)
                current_tokens = speech_token[:, token_st:token_et]
            else:
                print('token_offset==0 token:', token_offset, token_offset + window_length+3)
                # prompt_token[:, :10] 倾向于选静音的片段拼接，选到有声可能会有音频
                current_tokens = torch.cat((prompt_token[:, :10], speech_token[:, token_offset:token_offset + window_length+3]), dim=1)
                #speech_token[:, token_offset:token_offset + window_length+3]
                
            #print('current_tokens st:', token_offset, 'end:', token_offset + window_length+3)
            #pad_length = 0
            print('=====================no pad branch, available:', available)
            last_chunk=False
        else:
            # 需要填充
            #current_tokens = speech_token[:, token_offset:token_offset + available]
            print('=====================pad branch, available:', available)
            print('=====================token:', token_offset-10, token_offset + available)
            current_tokens = speech_token[:, token_offset-10:token_offset + available]
            pad_length = window_length + 3 - available
            token_pad = orginal_prompt_token[:, :pad_length]
            print('token_pad', token_pad.size(), 'current_tokens', current_tokens.size())
            current_tokens = torch.cat((current_tokens, token_pad), dim=1)
            print('current_tokens after pad', current_tokens.size())
            #, 'current_tokens', current_tokens.size())
            if pad_length >= token_hop_len or token_offset==0:
                last_chunk=True
            else:
                last_chunk=False
            
            # 在 tokens 维度（dim=1）向右填充 0
            #current_tokens = F.pad(current_tokens, (0, pad_length))
            print('ava current_tokens st:', token_offset, 'end:', token_offset + available+pad_length)

        # finalize=True 不需要context
        #flow_cache_dict = cosyvoice.model.init_flow_cache()
        flow_cache_dict = None
        #print('flow_cache_dict encoder_cache offset', flow_cache_dict['encoder_cache']['offset'])
        #print('flow_cache_dict decoder_cache offset', flow_cache_dict['decoder_cache']['offset'])
        tts_mel = token2wav(current_tokens, embedding, prompt_token, prompt_feat,
                        flow_cache_dict=flow_cache_dict,
                        hift_cache_dict={},
                        finalize=False,
                        trim_len=flow_decoder_required_cache_size, 
                        )
        print('prompt_token', prompt_token.shape, 'prompt_feat', prompt_feat.shape, 'current_tokens', current_tokens.shape)
        print('tts_mel', tts_mel.shape)
        
#         prompt_token = None
#         prompt_feat = None
        # 缓解stream 中部分ip音量忽大忽小的问题
        #tts_mel = tts_mel[:, :, :mel_end]
        prompt_feat = tts_mel[:, :, :mel_end].transpose(1,2)
        tts_mel_len = prompt_feat.shape[1]
        prompt_token = current_tokens[:, :tts_mel_len//token_mel_ratio]
        
        #speech_chunk, hift_cache_dict, speech_chunk_total = mel2wav_hift_cache(cosyvoice, tts_mel, hift_cache_dict=hift_cache_dict, last_chunk=last_chunk)
        speech_chunk = mel2wav_hift_nocache(cosyvoice, tts_mel)
        print('speech_chunk', speech_chunk.shape[1], speech_chunk.shape[1]/24000)
        #save_tmp_wav_path = save_wav_path.replace('.wav', '_%d.wav'%i)
        #torchaudio.save(save_tmp_wav_path, speech_chunk_tmp.detach().cpu(), sample_rate=24000)
        index = index+1
        
        #cur_chunk_len = speech_chunk.size(1)/24000.
        # syn_dur = syn_dur + cur_chunk_len
        # print('=======cur_chunk_len', cur_chunk_len, 'total syn len:', syn_dur)


        # 更新 token_offset
        token_offset += token_hop_len
        first_window = False
        #save_tmp1_path = save_wav_path.replace('.wav', '_slice_%d.wav'%index)
        
        # 终止条件：当 token_offset >= total_tokens 时退出循环
        if last_chunk and len(all_speech)==0:
            st = 0
            et = (10*total_tokens)*480
            print('cur speech ', st, et)
            speech_chunk = speech_chunk[:, st:et].cpu()
            #torchaudio.save(save_tmp1_path, speech_chunk.cpu(), sample_rate=24000) 
            all_speech = speech_chunk.numpy()
            break
        elif len(all_speech)==0:
            st = 10*token_mel_ratio*480
            et = (10*token_mel_ratio+mel_end)*480
            print('cur speech ', st, et)
            speech_chunk = speech_chunk[:, st:et].cpu()
            print('speech_chunk', speech_chunk.shape)
            #torchaudio.save(save_tmp1_path, speech_chunk.cpu(), sample_rate=24000) 
            all_speech = speech_chunk.numpy()
        else:
            if last_chunk:
                st = (10*token_mel_ratio-mel_cache_len)*480
                et = (10+available)*token_mel_ratio*480
            else: # 2~N-1 
                st = (10*token_mel_ratio-mel_cache_len)*480
                et = (10*token_mel_ratio+mel_end)*480
            print('cur speech ', st, et)
            speech_chunk = speech_chunk[:, st:et].cpu()
            print('speech_chunk', speech_chunk.shape)
            #torchaudio.save(save_tmp1_path, speech_chunk.cpu(), sample_rate=24000) 
        
            all_speech = crossfade_streaming(all_speech, speech_chunk.numpy(), speech_window)
        print('all_speech', all_speech.shape)
        if token_offset >= total_tokens:
            break
    # if len(all_speech)>1:
    #     last_speech_chunk = speech_chunk_total[:, :(available*token_mel_ratio+8)*480]
    #     last_speech_chunk_len = last_speech_chunk.size(1)/24000.
    #     print('last_speech_chunk_len:', last_speech_chunk_len)
    #     all_speech[-1] = last_speech_chunk.cpu()
    
    print('token_offset:', token_offset, 'total_tokens:', total_tokens)
    

    # 合并所有音频片段并保存
    all_speech = all_speech[:total_tokens*token_mel_ratio*480]
    all_speech = torch.from_numpy(all_speech.reshape(1,-1))
        
    print('syn_dur', syn_dur) 

    torchaudio.save(save_wav_path, all_speech, sample_rate=24000)  # 根据实际采样率调整
    print('save_wav_path:', save_wav_path)
    print('src_wav_dur:', src_wav_dur, 'syn_speech:', all_speech.size(1)/24000)
    #sys.exit()
    save_dict[os.path.basename(save_wav_path)] = {'total_tokens':total_tokens, 'available':available, 'pad_length': pad_length, 'src_wav_dur':src_wav_dur, 'syn_speech':syn_dur}
    return 


def fade_in_out(fade_in_mel, fade_out_mel, window):
    #print('fade_in_out fade_in_mel:', fade_in_mel.size(), 'fade_out_mel:', fade_out_mel.size())
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    #print('mel_overlap_len', mel_overlap_len)
    if fade_in_mel.device == torch.device('cpu'):
        fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    #print('fade_in_out fade_in_mel:', fade_in_mel.size())
    #sys.exit()
    return fade_in_mel.to(device)


def crossfade_streaming(audio1, audio2, overlap_length, sample_rate=24000):
    """
    对流式推理音频片段进行平滑拼接
    
    参数:
    audio1: 第一段音频 (numpy数组)
    audio2: 第二段音频 (numpy数组)
    overlap_length: 重叠部分的长度(样本点数)
    sample_rate: 采样率 (默认24000)
    
    返回:
    拼接后的完整音频 (numpy数组)
    """
    
    if audio1.shape[0] ==1:
        audio1 = audio1.squeeze(0)
    if audio2.shape[0] ==1:
        audio2 = audio2.squeeze(0)
    print('crossfade_streaming', audio1.shape, audio2.shape)
    # 1. 验证输入参数
    if overlap_length < 0:
        raise ValueError("重叠长度必须大于0")
    elif overlap_length==0:
        print("~~~~~~~~~~~~~~~~~重叠长度==0")
        result = np.concatenate([audio1, audio2])
        return result
    if len(audio1) < overlap_length or len(audio2) < overlap_length:
        print("警告: 音频片段长度不足，无法完全处理重叠部分")
        sys.exit()
    else:
        actual_overlap = overlap_length
    
    # 2. 创建交叉淡化窗口
    # 使用改进的余弦函数窗口，实现更平滑的过渡
    t = np.linspace(0, np.pi, actual_overlap)
    fade_out = np.sqrt(0.5 * (1 + np.cos(t)))  # 淡出窗口 (前一段)
    fade_in = np.sqrt(0.5 * (1 - np.cos(t)))   # 淡入窗口 (后一段)
    
    # 3. 提取重叠部分
    end_of_audio1 = audio1[-actual_overlap:]
    start_of_audio2 = audio2[:actual_overlap]
    
    # 4. 应用交叉淡化
    mixed = end_of_audio1 * fade_out + start_of_audio2 * fade_in
    
    # 5. 拼接音频
    result = np.concatenate([
        audio1[:-actual_overlap],  # 第一段音频的非重叠部分
        mixed,                     # 混合后的重叠部分
        audio2[actual_overlap:]    # 第二段音频的非重叠部分
    ])
    
    return result


def tts_stream_generator_melovel(prompt_audio_path, save_wav_path, vq_path, device, prompt_token_path=None, token_hop_len=25, sub_hop_len=0, pre_lookahead_len=3, token_mel_ratio=1):
    
    if token_mel_ratio==1:
        speech_token, embedding, src_wav_dur, _ = extract_speech_token_emb(prompt_audio_path, vq_path, device, token_mel_ratio=token_mel_ratio)
        prompt_feat = torch.zeros(1, 0, 80).to(device)
        prompt_token=torch.zeros(1, 0, dtype=torch.int32).to(device)
    elif token_mel_ratio==2:
        speech_token, embedding, src_wav_dur, _, prompt_token, prompt_feat = extract_token25_emb_mel(prompt_audio_path, vq_path, device, with_pad=False, prompt_token_path=prompt_token_path, token_mel_ratio=token_mel_ratio, first_hop_len=token_hop_len, sub_hop_len=sub_hop_len)
        init_prompt_feat_len = prompt_feat.shape[1]
    orginal_prompt_token = prompt_token
        
        


    print('speech_token', speech_token.size())
    flow_cache_dict = cosyvoice.model.init_flow_cache()
    hift_cache_dict = None
    #flow_decoder_required_cache_size = self.flow.decoder.estimator.num_decoding_left_chunks * self.flow.decoder.estimator.static_chunk_size
    flow_decoder_required_cache_size = int(1* 0.5*50*1)
    # estimator.num_decoding_left_chunks = default=1/ 0.5
    # estimator.static_chunk_size = <chunk_size> * <token_frame_rate> * <token_mel_ratio>
        
    total_tokens = speech_token.size(1) #b,t, dim len(tts_speech_tokens)
    token_offset = 0
    all_speech = []

    mel_cache_len = 2
    speech_window = mel_cache_len*480 #np.hamming(2 * mel_cache_len*480)
                               
    first_window = True
    pre_tts_mel = None
    
    token_hop_len = 50
    mel_end = token_hop_len * token_mel_ratio
    pre_lookahead_len = 15
    index = 0
    syn_dur = 0
    while True:
        
        window_length = token_hop_len + pre_lookahead_len
        available = total_tokens - token_offset

        if available >= window_length+3:
            # 正常提取
            current_tokens = speech_token[:, token_offset:token_offset + window_length+3]
            print('current_tokens st:', token_offset, 'end:', token_offset + window_length+3)
            #pad_length = 0
            print('=====================no pad branch, available:', available)
            last_chunk=False
        else:
            # 需要填充
            current_tokens = speech_token[:, token_offset:token_offset + available]
            pad_length = window_length + 3 - available
            token_pad = orginal_prompt_token[:, :pad_length]
            print('token_pad', token_pad.size(), 'current_tokens', current_tokens.size())
            current_tokens = torch.cat((current_tokens, token_pad), dim=1)
            print('current_tokens after pad', current_tokens.size())
            #, 'current_tokens', current_tokens.size())
            if pad_length >= token_hop_len or token_offset==0:
                last_chunk=True
            else:
                last_chunk=False
            print('=====================pad branch, available:', available)
            # 在 tokens 维度（dim=1）向右填充 0
            #current_tokens = F.pad(current_tokens, (0, pad_length))
            print('ava current_tokens st:', token_offset, 'end:', token_offset + available+pad_length)

        # finalize=True 不需要context
        flow_cache_dict = cosyvoice.model.init_flow_cache()
        #print('flow_cache_dict encoder_cache offset', flow_cache_dict['encoder_cache']['offset'])
        #print('flow_cache_dict decoder_cache offset', flow_cache_dict['decoder_cache']['offset'])
        tts_mel = token2wav(current_tokens, embedding, prompt_token, prompt_feat,
                        flow_cache_dict=flow_cache_dict,
                        hift_cache_dict={},
                        finalize=False,
                        trim_len=flow_decoder_required_cache_size,  

                        )
        
        #  prompt_token = None
        #  prompt_feat = None
        # 缓解stream 中部分ip音量忽大忽小的问题
        tts_mel = tts_mel[:, :, :mel_end]
        tts_mel_concat = torch.cat((prompt_feat.transpose(1,2), tts_mel), dim=2)
        prompt_feat = tts_mel.transpose(1,2)
        tts_mel_len = prompt_feat.shape[1]
        prompt_token = current_tokens[:, :tts_mel_len//token_mel_ratio]
        print('tts_mel', tts_mel.shape)
        
        print('tts_mel_concat', tts_mel_concat.shape)

        #tts_mel torch.Size([1, 80, 100])
        #tts_mel_concat torch.Size([1, 80, 200])

        speech_chunk_concat = mel2wav_hift_nocache(cosyvoice, tts_mel_concat)
        save_tmp1_path = save_wav_path.replace('.wav', '_concat_%d.wav'%index)
        #torchaudio.save(save_tmp1_path, speech_chunk_concat.cpu(), sample_rate=24000) 
        print('speech_chunk_concat', speech_chunk_concat.shape) 
        #speech_chunk_concat torch.Size([1, 96000])
        #speech_chunk = speech_chunk_concat[:, tts_mel_len*480:]
        
        
        
        
        
        
        save_tmp_path = save_wav_path.replace('.wav', '_%d.wav'%index)
        
        # 更新 token_offset
        token_offset += token_hop_len
        first_window = False
        if len(all_speech)>=1:
            speech_chunk = speech_chunk_concat[:, (tts_mel_len-8)*480:]
            
            #torchaudio.save(save_tmp_path, speech_chunk.cpu(), sample_rate=24000) 
            all_speech = crossfade_streaming(all_speech, speech_chunk.cpu().numpy(), speech_window)
        else:
            #all_speech.append(speech_chunk[:, 8*480:].cpu())
            speech_chunk = speech_chunk_concat
            #torchaudio.save(save_tmp_path, speech_chunk.cpu(), sample_rate=24000) 
            all_speech = speech_chunk[:, init_prompt_feat_len*480:].cpu().numpy()
            
            
        # 终止条件：当 token_offset >= total_tokens 时退出循环
        #all_speech.append(speech_chunk.cpu())
        index = index+1
        if token_offset >= total_tokens:
            break
    
    # save_tmp_path = save_wav_path.replace('.wav', '_%d.wav'%index)
    # if len(all_speech)>1:
    #     speech_chunk = speech_chunk_concat[:, (tts_mel_len-8)*480:(tts_mel_len+available*token_mel_ratio)*480]
    #     torchaudio.save(save_tmp_path, speech_chunk.cpu(), sample_rate=24000) 
    #     all_speech = crossfade_streaming(all_speech, speech_chunk.cpu().numpy(), speech_window)
    # else:
    #     all_speech = speech_chunk[:, init_prompt_feat_len*480:].cpu().numpy()
    
    print('token_offset:', token_offset, 'total_tokens:', total_tokens)
    
    

    # 合并所有音频片段并保存
    syn_dur = 0

    print('syn_dur', syn_dur) 
    all_speech = all_speech.reshape(1,-1)
    full_speech = torch.from_numpy(all_speech)
    #torch.cat(all_speech, dim=1)
    torchaudio.save(save_wav_path, full_speech, sample_rate=24000)  # 根据实际采样率调整
    print('save_wav_path:', save_wav_path)
    print('src_wav_dur:', src_wav_dur, 'syn_speech:', full_speech.size(1)/24000)
    #sys.exit()
    save_dict[os.path.basename(save_wav_path)] = {'total_tokens':total_tokens, 'available':available, 'pad_length': pad_length, 'src_wav_dur':src_wav_dur, 'syn_speech':syn_dur}
    return 



    

           
                                             
    

def cosy_flow_token2wav_stream(prompt_audio_path, device, sr=16000):
    wav = load_wav(prompt_audio_path, sr) # [1, T]
    gain_db = -24 - math.log10(torch.mean(wav * wav).cpu().item() + 1e-16) * 10
    wav = torchaudio.functional.gain(wav, gain_db)
    wav = torch.clip(wav, min=-1, max=1)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    speech_token, speech_token_len = extract_speech_token(speech_tokenizer_session, wav, device)
    embedding = extract_spk_embedding(wav, device)
    
    #prompt_feat=torch.zeros(1, 0, 80).to(device)
    #prompt_token=torch.zeros(1, 0, dtype=torch.int32).to(device),
    
        
    tts_speech = cosyvoice.model.flow_token2wav(speech_token, embedding, stream=True)
    print('tts_speech', tts_speech)
    
def copy_gan(prompt_audio_path, device, sr=16000):
    # torch.audio [-1,1]
    wav = load_wav(prompt_audio_path, sr) # [1, T]
    #.squeeze(0) 
    print('=====wav', wav.size())
    #sr, audio = prompt_audio
    #wav = torch.tensor(audio)
    #wav = wav.to(device) / 32767.0
    #if wav.ndim == 2:
    #   print("=========wav double channel into single")
    #    wav = wav.mean(-1)
    #wav = wav.unsqueeze(0)
    #print("=========wav unsqueeze", wav.size())
    gain_db = -24 - math.log10(torch.mean(wav * wav).cpu().item() + 1e-16) * 10
    wav = torchaudio.functional.gain(wav, gain_db)
    wav = torch.clip(wav, min=-1, max=1)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    speech_token, speech_token_len = extract_speech_token(speech_tokenizer_session, wav, device)
    embedding = extract_spk_embedding(wav, device)
    print('speech_token', speech_token.size()) #, speech_token.device)
    print('embedding', embedding.size()) #, embedding.device)
    # 
    tts_mel = cosyvoice.model.flow.inference(token=speech_token,
                                             token_len=speech_token_len,
                                             prompt_token=torch.zeros(1, 0, dtype=torch.int32).to(device),
                                             prompt_token_len=torch.zeros(1, dtype=torch.int32).to(device),
                                             prompt_feat=torch.zeros(1, 0, 80).to(device),
                                             prompt_feat_len=torch.zeros(1, dtype=torch.int32).to(device),
                                             embedding=embedding)
    tts_speech, tts_source = cosyvoice.model.hift.inference(mel=tts_mel)
    print('tts_speech', tts_speech)
    print('tts_speech', tts_speech.size(), tts_speech)
    tts_speech = tts_speech.cpu()
    #wav = (tts_speech * 32767).astype(np.int16)
    # epoch_0_step_50000.pt
    
    infer_model_tag = os.path.basename(args.flow_path).replace('.pt', '')
    wav_name = os.path.basename(prompt_audio_path).replace('.wav', '')
    tts_wav_path = os.path.join('model_outputs/24k_new/wav/', '%s_%s.wav'%(infer_model_tag, wav_name))
    os.makedirs(os.path.dirname(tts_wav_path), exist_ok=True)
    #tts_fn = 'save_self_22k.wav'
    save_sr = 22050
    
    torchaudio.save(tts_wav_path, tts_speech, save_sr)
    print('saved audio to ', tts_wav_path)
    return #22100, wav[0]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('unsupported value encontoured.')

        
def fm_inference(prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio, stream, prompt_token_path):
    if stream:
        #tts_stream_generator_melovel(prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio=token_mel_ratio, sub_hop_len=0, prompt_token_path=prompt_token_path)
        tts_stream_generator2(prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio=token_mel_ratio, sub_hop_len=0, prompt_token_path=prompt_token_path)
    else:
        tts_stream_generator_full(prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio=token_mel_ratio, prompt_token_path=prompt_token_path)
            
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_path',
                        type=str,
          default= 'CosyVoice/pretrained_models/flow.pt',
                        help='local path or modelscope repo id')
    parser.add_argument('--token_rate',
                        type=str,
                        help='50hz or 25hz', default='25hz')
    parser.add_argument('--wav_dir',
                        type=str,
                        default='01',
                        help='local path or modelscope repo id')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--vq_dir', default='xx')
    parser.add_argument('--stream', type=str2bool, default=1)
                       
    args = parser.parse_args()
    model_dir = 'CosyVoice/pretrained_models'
    
    if args.token_rate=='50hz':
        flow_path = 'cosyvoice1/llm.pt'
        token_mel_ratio = 1
    elif args.token_rate=='25hz':
        flow_path = args.flow_path
        
        token_mel_ratio = 2

    config_path=''
    cosyvoice = CosyVoice2(model_dir, flow_path=flow_path, config_path=config_path, fp16=False)
    
    print('load CosyVoice model finished')
    print('=======token_mel_ratio:', token_mel_ratio)
    #sys.exit() 
    #infer_model_tag = os.path.basename(args.flow_path).replace('.pt', '')
    #flow_ckpt_name = os.path.basename(os.path.dirname(os.path.dirname(args.flow_path)))
    
    os.makedirs(args.save_dir, exist_ok=True)
    #tts_mel_path = 'model_outputs/24k_new/npy/01.npy'

    n_timesteps = 10
    


    # target_wav_path|save_name|zh|prompt_wav_path
    '''
    if os.path.isfile(args.wav_dir):
        test_lines = open(args.wav_dir).readlines()
        for line in test_lines:
            target_wav_path, save_name, _, prompt_wav_path = line.strip().split('|')
            spk = os.path.basename(os.path.dirname(target_wav_path))
            vq_path = os.path.join(args.vq_dir, spk, '%s.npy'%os.path.basename(target_wav_path).split('.')[0])
            tts_wav_path = os.path.join(args.save_dir, '%s.wav'%save_name)
            fm_inference(prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio, stream)


    
    # seed_tts: seed_tts/meta_all.lst
    # file format : target_wav_path|prompt_txt|prompt_wav_path|target_txt
    if os.path.isfile(args.wav_dir):
        test_lines = open(args.wav_dir).readlines()
        for line in test_lines:
            target_wav_path, prompt_txt, prompt_wav_path, target_txt = line.strip().split('|')
            
            
            save_name = os.path.basename(target_wav_path).split('.')[0]
            vq_path = os.path.join(args.vq_dir, '%s.npy'%save_name)
            tts_wav_path = os.path.join(args.save_dir, '%s.wav'%save_name)
            #fm_inference(prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio, args.stream)
            prompt_name = os.path.basename(prompt_wav_path).split('.')[0]
            prompt_token_path = os.path.join(args.vq_dir, '%s.npy'%prompt_name)
            fm_inference(prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio, args.stream, prompt_token_path)
            
    '''
    # ar + fm
    print(sys.path)
    # token_dir: ar outputs, dim:[1, T]
    token_dir = ''

    # prompt_token_path: dim:[T, 1]

    prompt_token_path = 'xx.npy'
    prompt_wav_path = 'xx.wav'

    
    token_list = os.listdir(token_dir)
    token_list = [i for i in token_list if i.endswith('.npy')]
    '''
    for fi in token_list[:1]:
        vq_path = os.path.join(token_dir, fi)
        
        prompt_token_path = prompt_token_path     
        tts_wav_path = os.path.join(args.save_dir, fi.replace('.npy', '.wav'))
        
        
    
        # (prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio, stream, prompt_token_path
        if token_mel_ratio == 2:
            fm_inference(prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio, args.stream, prompt_token_path)
        elif token_mel_ratio == 1:
            fm_inference(prompt_wav_path, tts_wav_path, vq_path, device, token_mel_ratio, args.stream, None)
    print(save_dict)
        #sys.exit()
    '''
    tts_mel = torch.from_numpy(np.load('test_mel.npy')).to(device)
    print('tts_mel', tts_mel.shape)
    tts_speech = mel2wav_hift_nocache(cosyvoice, tts_mel)
    audio = tts_speech.detach().cpu()
    print('audio', audio.shape)
    tts_audio_path = 'test_mel.wav'
    torchaudio.save(tts_audio_path, audio, 24000)
    print('mel2wav test_mel.wav')

