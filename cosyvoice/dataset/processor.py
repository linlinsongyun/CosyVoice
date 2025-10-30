# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import List
import random
import sys
import librosa
import numpy as np
import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import pyworld as pw  
from cosyvoice.utils.file_utils import load_wav
torchaudio.set_audio_backend('soundfile')

AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}

import struct
from protos.text_data_pb2 import TextData
def pb_opener(data, mode='train', tts_data={}):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            with open(url, 'rb') as f:
                while True:
                    d = f.read(4)
                    if len(d) != 4:
                        break
                    length = struct.unpack('I', d)[0]
                    data = f.read(length)
                    text_data = TextData()
                    text_data.ParseFromString(data)
                    samples = list(text_data.sentences)
                    random.shuffle(samples)
                    for sentence in samples:
                        
                        sample["utt"] = sentence.file_path
                        sample["wav"] = sentence.file_path
                        try:
                            if os.path.isfile(sentence.file_path):
                                pass
                            else:
                                sentence.file_path = sentence.file_path.replace('/wavs/', '/wavs_concat_6s/')
                            #wavs_concat_6s
                            with open(sentence.file_path, 'rb') as af:
                                sample["audio_data"] = af.read()
                        except Exception as ex:
                            #logging.warning('read file {} failed, ex info {}'.format(sentence.file_path, ex))
                            continue
                        sample["text"] = sentence.text
                        sample["spk"] = text_data.name
                        sample["utt_embedding"] = list(sentence.emb)
                        sample["spk_embedding"] = list(sentence.emb)
                        sample["speech_token"] = list(sentence.semantics)
                        sample["wav_path"] = str(sentence.file_path)
                        yield {**sample}

        except Exception as ex:
            logging.warning('Failed to open {}, ex info {}'.format(url, ex))

           
        
def parquet_opener(data, mode='train', tts_data={}):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            for df in pq.ParquetFile(url).iter_batches(batch_size=64):
                df = df.to_pandas()
                for i in range(len(df)):
                    if mode == 'inference' and df.loc[i, 'utt'] not in tts_data:
                        continue
                    sample.update(dict(df.loc[i]))
                    if mode == 'train':
                        # NOTE do not return sample directly, must initialize a new dict
                        yield {**sample}
                    else:
                        for index, text in enumerate(tts_data[df.loc[i, 'utt']]):
                            yield {**sample, 'tts_index': index, 'tts_text': text}
        except Exception as ex:
            logging.warning('Failed to open {}, ex info {}'.format(url, ex))


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        sample['speech'], sample['sample_rate'] = torchaudio.load(BytesIO(sample['audio_data']))
        sample['speech'] = sample['speech'].mean(dim=0, keepdim=True)
        del sample['audio_data']
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['speech'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample['text_token']) < token_min_length:
            continue
        if len(sample['text_token']) > token_max_length:
            continue
        if len(sample['speech_token']) == 0:
            continue
        if num_frames != 0:
            if len(sample['text_token']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['text_token']) / num_frames > max_output_input_ratio:
                continue
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode='train'):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['speech']
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                continue
            sample['sample_rate'] = resample_rate
            sample['speech'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        max_val = sample['speech'].abs().max()
        if max_val > 1:
            sample['speech'] /= max_val
        yield sample


def truncate(data, truncate_length=24576, mode='train'):
    """ Truncate data.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            truncate_length: truncate length

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        waveform = sample['speech']
        if waveform.shape[1] > truncate_length:
            start = random.randint(0, waveform.shape[1] - truncate_length)
            waveform = waveform[:, start: start + truncate_length]
            sample['truncate_st'] = start
        else:
            waveform = torch.concat([waveform, torch.zeros(1, truncate_length - waveform.shape[1])], dim=1)
            sample['truncate_st'] = 0
        sample['speech'] = waveform
        
        yield sample

def compute_fbank_truncate(data,
                  feat_extractor,
                  token_mel_ratio=2,
                  mode='train'):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        #print('===================compute_fbank_truncate sample===================')
        # for key in sample:
        #     print(key)
        # sys.exit()
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        #print('feat_extractor mel', mat.shape)
        #print('speech_token 25hz', len(sample['speech_token']))

        sample['speech_feat'] = mat
        #print('wavform ', sample['speech'].shape)
        #print('speech_feat ', sample['speech_feat'].shape)

        yield sample

        
def compute_fbank(data,
                  feat_extractor,
                  token_mel_ratio=2,
                  mode='train'):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        #print('===================compute_fbank sample===================')
        # for key in sample:
        #     print(key)
        # sys.exit()
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        #print('feat_extractor mel', mat.shape)
        #print('speech_token 25hz', len(sample['speech_token']))

        #speech_feat torch.Size([861, 160])
        #speech_token 432
        # cosy2: 
        speech_token_len = len(sample['speech_token'])
        mat_len = speech_token_len * token_mel_ratio
        #print('orignal speech_token_len', speech_token_len)
        #print('orignal speech_feat', mat.shape)
        
        if mat_len > mat.size(0):
            pad_mel_len = speech_token_len * token_mel_ratio - mat.size()[0]
            #print('pad_mel_len', pad_mel_len)
        
            # 获取最后一个元素作为 padding 值
            padding_value = mat[-1, :].unsqueeze(0).expand(pad_mel_len, -1)  # 将最后一行复制三次

            # 直接拼接到原始 tensor 上
            new_mat = torch.cat((mat, padding_value), dim=0)
        else:
            # mat: [T,80]
            new_mat = mat[:mat_len, :]
            #print('compute_fbank', new_mat.size())
        #print('speech_feat mat', mat.size(), 'pad_mel_len', pad_mel_len, 'speech_token_len', speech_token_len, 'new_mat', new_mat.size())
        #print('speech_feat new_mat', new_mat.size())
        #print('speech_token', len(sample['speech_token']))
        #sys.exit()
        sample['speech_feat'] = new_mat
        #save_path = os.path.join('01', os.path.basename(sample["wav_path"]).replace('.wav', '.npy'))
        #np.save(save_path, mat.cpu().numpy())
        #print('=================== wav_path', sample["wav_path"])
        #print('length same process mel:', new_mat.shape)
        #sys.exit()
        #print('wavform ', sample['speech'].shape)
        #print('speech_feat ', sample['speech_feat'].shape)
        yield sample


# def compute_f0(data, pitch_extractor, mode='train'):
#     """ Extract f0

#         Args:
#             data: Iterable[{key, wav, label, sample_rate}]

#         Returns:
#             Iterable[{key, feat, label}]
#     """
#     for sample in data:
#         print('===================compute_f0 sample===================')
#         for key in sample:
#             print(key)
#         # assert 'sample_rate' in sample
#         # assert 'speech' in sample
#         # assert 'utt' in sample
#         # assert 'text_token' in sample
#         waveform = sample['speech']
#         mat = pitch_extractor(waveform).transpose(1, 2)
#         mat = F.interpolate(mat, size=sample['speech_feat'].shape[0], mode='linear')
#         sample['pitch_feat'] = mat[0, 0]
#         yield sample

def compute_f0(data, sample_rate, hop_size, mode='train'):
    """ Extract f0

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    frame_period = hop_size * 1000 / sample_rate
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        #waveform = sample['speech']
        f0_basedir = '/mnt/nas1/zhangying/cosy_data_25hz/sft_data/f0_data'
        f0_path = sample["wav_path"].replace('/mnt/nas1', f0_basedir).replace('.wav', '.npy')
        # 不能预先存储，因为随机截取的wav 片段
        
        truncate_st = sample['truncate_st']
        st_f0 = int(truncate_st/hop_size)
        #et_f0 = int(truncate_et/hop_size)
        f0_len = sample['speech_feat'].shape[0] 
        et_f0 = st_f0+f0_len
        try:
            #print('f0_path', f0_path)
            f0 = np.load(f0_path)


        except: 
            os.makedirs(os.path.dirname(f0_path), exist_ok=True)
        #sample['truncate']
            waveform = load_wav(sample["wav_path"], 24000)

        
            _f0, t = pw.harvest(waveform.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period)
            if sum(_f0 != 0) < 5: # this happens when the algorithm fails
                _f0, t = pw.dio(waveform.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period) # if harvest fails, try dio
            f0 = pw.stonemask(waveform.squeeze(dim=0).numpy().astype('double'), _f0, t, sample_rate) # f0: len(wav)/hop_size
            np.save(f0_path, f0)
            
        if len(f0) >= f0_len:
            pass
        else:
            f0 = np.concatenate((f0, [0]*(f0_len-len(f0))), axis=0)

        try:
            f0_data = F.interpolate(torch.from_numpy(f0).view(1, 1, -1), size=f0_len, mode='linear').view(-1)
        except: 
            
            
            print(sample["wav_path"], len(f0_ori), st_f0, et_f0, 'slice_f0_len', f0_len, 'truncate_st:', truncate_st, 'hop_size:', hop_size)
        sample['pitch_feat'] = f0_data
        # print('=============compute_f0 sample_rate', sample['sample_rate'])
        # print('hop_size', hop_size)
        # print('=============compute_f0 speech', sample['speech'].shape)
        # print('=============compute_f0 pitch_feat', f0.shape)
        # print('=============compute_f0 speech_feat', sample['speech_feat'].shape)
        yield sample
        

def parse_embedding(data, normalize, mode='train'):
    """ Parse utt_embedding/spk_embedding

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        sample['utt_embedding'] = torch.tensor(sample['utt_embedding'], dtype=torch.float32)
        sample['spk_embedding'] = torch.tensor(sample['spk_embedding'], dtype=torch.float32)
        if normalize:
            sample['utt_embedding'] = F.normalize(sample['utt_embedding'], dim=0)
            sample['spk_embedding'] = F.normalize(sample['spk_embedding'], dim=0)
        yield sample


def tokenize(data, get_tokenizer, allowed_special, mode='train'):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    for sample in data:
        assert 'text' in sample
        sample['text_token'] = tokenizer.encode(sample['text'], allowed_special=allowed_special)
        if mode == 'inference':
            sample['tts_text_token'] = tokenizer.encode(sample['tts_text'], allowed_special=allowed_special)
        yield sample


def shuffle(data, shuffle_size=10000, mode='train'):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['speech_feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['speech_feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'speech_feat' in sample
        assert isinstance(sample['speech_feat'], torch.Tensor)
        new_sample_frames = sample['speech_feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    """ Wrapper for static/dynamic batch
    """
    if mode == 'inference':
        return static_batch(data, 1)
    else:
        if batch_type == 'static':
            return static_batch(data, batch_size)
        elif batch_type == 'dynamic':
            return dynamic_batch(data, max_frames_in_batch)
        else:
            logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data, use_spk_embedding, mode='train', gan=False):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        speech_feat_len = torch.tensor([x['speech_feat'].size(1) for x in sample],
                                       dtype=torch.int32)
        order = torch.argsort(speech_feat_len, descending=True)

        utts = [sample[i]['utt'] for i in order]
        speech = [sample[i]['speech'].squeeze(dim=0) for i in order]
        speech_len = torch.tensor([i.size(0) for i in speech], dtype=torch.int32)
        speech = pad_sequence(speech, batch_first=True, padding_value=0)
        speech_token = [torch.tensor(sample[i]['speech_token']) for i in order]
        speech_token_len = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)
        speech_token = pad_sequence(speech_token,
                                    batch_first=True,
                                    padding_value=0)
        speech_feat = [sample[i]['speech_feat'] for i in order]
        speech_feat_len = torch.tensor([i.size(0) for i in speech_feat], dtype=torch.int32)
        
        speech_feat = pad_sequence(speech_feat,
                                       batch_first=True,
                                       padding_value=0)
        #except:
        #    original_lengths = [feat.size(0) for feat in speech_feat]
        #    print("原始样本长度:", original_lengths)
        #    original_dims = [feat.size(1) for feat in speech_feat]
        #    print("填充后统一长度:", original_dims)

            
        text = [sample[i]['text'] for i in order]
        text_token = [torch.tensor(sample[i]['text_token']) for i in order]
        text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
        utt_embedding = torch.stack([sample[i]['utt_embedding'] for i in order], dim=0)
        spk_embedding = torch.stack([sample[i]['spk_embedding'] for i in order], dim=0)
        batch = {
            "utts": utts,
            "speech": speech,
            "speech_len": speech_len,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_len,
            "text": text,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "utt_embedding": utt_embedding,
            "spk_embedding": spk_embedding,
        }
        if gan is True:
            # in gan train, we need pitch_feat
            pitch_feat = [sample[i]['pitch_feat'] for i in order]
            pitch_feat_len = torch.tensor([i.size(0) for i in pitch_feat], dtype=torch.int32)
            pitch_feat = pad_sequence(pitch_feat,
                                      batch_first=True,
                                      padding_value=0)
            batch["pitch_feat"] = pitch_feat
            batch["pitch_feat_len"] = pitch_feat_len
        else:
            # only gan train needs speech, delete it to save memory
            del batch["speech"]
            del batch["speech_len"]
        if mode == 'inference':
            tts_text = [sample[i]['tts_text'] for i in order]
            tts_index = [sample[i]['tts_index'] for i in order]
            tts_text_token = [torch.tensor(sample[i]['tts_text_token']) for i in order]
            tts_text_token_len = torch.tensor([i.size(0) for i in tts_text_token], dtype=torch.int32)
            tts_text_token = pad_sequence(tts_text_token, batch_first=True, padding_value=-1)
            batch.update({'tts_text': tts_text,
                          'tts_index': tts_index,
                          'tts_text_token': tts_text_token,
                          'tts_text_token_len': tts_text_token_len})
        if use_spk_embedding is True:
            batch["embedding"] = batch["spk_embedding"]
        else:
            batch["embedding"] = batch["utt_embedding"]
        yield batch

      
def pad_or_truncate_list(tensor_list: List[torch.Tensor], target_length: int, padding_value: float = 0.0) -> torch.Tensor:
    """
    将一个张量列表中的每个张量填充或截断到指定长度，然后堆叠成一个批处理张量。
    """
    processed_list = []
    for tensor in tensor_list:
        current_length = tensor.shape[-1]
        
        if current_length > target_length:
            # 截断
            processed_tensor = tensor[..., :target_length]
        elif current_length < target_length:
            # 填充
            pad_needed = target_length - current_length
            processed_tensor = F.pad(tensor, (0, pad_needed), 'constant', padding_value)
        else:
            processed_tensor = tensor
            
        processed_list.append(processed_tensor)
        
    return torch.stack(processed_list, dim=0)
   
def padding_with_audio_samples(data, use_spk_embedding, mode='train', gan=False, n_times=960):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    
    for sample in data:
        assert isinstance(sample, list)
        speech_feat_len = torch.tensor([x['speech_feat'].size(1) for x in sample],
                                       dtype=torch.int32)
        order = torch.argsort(speech_feat_len, descending=True) # 从大到小的index

        utts = [sample[i]['utt'] for i in order]
        speech = [sample[i]['speech'].squeeze(dim=0) for i in order]
        speech_len = torch.tensor([i.size(0) for i in speech], dtype=torch.int32)
        max_speech_len = max(speech_len)
        pad_data_len = n_times - max_speech_len%n_times
        max_ntimes_speech_len = max_speech_len + pad_data_len
        speech = pad_or_truncate_list(speech, max_ntimes_speech_len)
        #speech = pad_sequence(speech, batch_first=True, padding_value=0)
        
        speech_token = [torch.tensor(sample[i]['speech_token']) for i in order]
        #print('speech_token', speech_token[0].shape)
        speech_token_len = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)
        max_token_len = max(speech_token_len)
        expected_len_from_speech = max_ntimes_speech_len // 960
        # wav_pad 之后会比token 长
        if max_token_len != expected_len_from_speech:
            padding_tensor = torch.tensor([0], device=speech_token[0].device, dtype=speech_token[0].dtype)
            speech_token[0] = torch.cat((speech_token[0], padding_tensor), dim=0)
            speech_token_len[0] = speech_token_len[0]+1
            
        speech_token = pad_sequence(speech_token,
                                    batch_first=True,
                                    padding_value=0)
        



        speech_feat = [sample[i]['speech_feat'] for i in order]
        speech_feat_len = torch.tensor([i.size(0) for i in speech_feat], dtype=torch.int32)
        
        speech_feat = pad_sequence(speech_feat,
                                       batch_first=True,
                                       padding_value=0)
        #except:
        #    original_lengths = [feat.size(0) for feat in speech_feat]
        #    print("原始样本长度:", original_lengths)
        #    original_dims = [feat.size(1) for feat in speech_feat]
        #    print("填充后统一长度:", original_dims)

            
        text = [sample[i]['text'] for i in order]
        text_token = [torch.tensor(sample[i]['text_token']) for i in order]
        text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
        utt_embedding = torch.stack([sample[i]['utt_embedding'] for i in order], dim=0)
        spk_embedding = torch.stack([sample[i]['spk_embedding'] for i in order], dim=0)
        batch = {
            "utts": utts,
            "speech": speech,
            "speech_len": speech_len,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_len,
            "text": text,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "utt_embedding": utt_embedding,
            "spk_embedding": spk_embedding,
        }
        #print('=========================padding_with_audio_samples')
        #utts
        # speech_token
        # speech_token_len
        # speech_feat
        # speech_feat_len
        # text
        # text_token
        # text_token_len
        # utt_embedding
        # spk_embedding
        # embedding

        if gan is True:
            # in gan train, we need pitch_feat
            pitch_feat = [sample[i]['pitch_feat'] for i in order]
            pitch_feat_len = torch.tensor([i.size(0) for i in pitch_feat], dtype=torch.int32)
            pitch_feat = pad_sequence(pitch_feat,
                                      batch_first=True,
                                      padding_value=0)
            batch["pitch_feat"] = pitch_feat
            batch["pitch_feat_len"] = pitch_feat_len
        else:
            # only gan train needs speech, delete it to save memory
            #=============speech_feat torch.Size([2, 188, 80])
            #=============padding_with_audio_samples speech[0] torch.Size([181104])

            #print('=============speech_feat', batch["speech_feat"].shape)
            pass
            # vae del
            # del batch["speech_feat"]
            # del batch["speech_feat_len"]
            #print('=============padding_with_audio_samples speech[0]', speech[0].shape)
            # pad speech : torch.Size([2, 181104])
            #print('pad speech :', batch["speech"].shape)

            
        if mode == 'inference':
            tts_text = [sample[i]['tts_text'] for i in order]
            tts_index = [sample[i]['tts_index'] for i in order]
            tts_text_token = [torch.tensor(sample[i]['tts_text_token']) for i in order]
            tts_text_token_len = torch.tensor([i.size(0) for i in tts_text_token], dtype=torch.int32)
            tts_text_token = pad_sequence(tts_text_token, batch_first=True, padding_value=-1)
            batch.update({'tts_text': tts_text,
                          'tts_index': tts_index,
                          'tts_text_token': tts_text_token,
                          'tts_text_token_len': tts_text_token_len})
        if use_spk_embedding is True:
            batch["embedding"] = batch["spk_embedding"]
        else:
            batch["embedding"] = batch["utt_embedding"]
        yield batch
