from protos.text_data_pb2 import TextData
import struct
from pathlib import Path
import os
import numpy as np
import librosa

token_hz = 25
def read_proto_duration(p):
    total_semantic_token_num = 0
    total_duration = 0
    wav_root = 'dir'
    emb_root = 'base_emb' 
    with open(p, 'rb') as f:
        while True:
            d = f.read(4)
            # print(d)
            if len(d) != 4:
                # print(len(d))
                break
            length = struct.unpack('I', d)[0]
            #print('length', length)
            data = f.read(length)
            text_data = TextData()
            text_data.ParseFromString(data)
            
            cnt = len(text_data.sentences)
            samples = list(text_data.sentences)


            while len(samples):
                sentence = samples.pop()
                total_semantic_token_num += len(sentence.semantics)
                
    #print(f'{total_semantic_token_num/25/60/60} h')
    tmp_dur = total_semantic_token_num/25/60/60
    return tmp_dur

protos_dir = 'split_protos_v2/'



from pathlib import Path
root_path = Path(protos_dir)
total_dur = 0
proto_lists = list(root_path.rglob('*.proto')) + list(root_path.rglob('*.protos'))
for proto_file in proto_lists:
    
    tmp_dur = read_proto_duration(proto_file)
    print(proto_file, tmp_dur)
    total_dur += tmp_dur
    
print('total_dur:', total_dur)
