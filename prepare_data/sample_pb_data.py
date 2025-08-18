import torch.cuda
from protos.text_data_pb2 import TextData
print('TextData', TextData)

import sys
import struct
import os
import random
import numpy as np
#from encodec import EncodecModel
import torchaudio

device = 'cuda' if torch.cuda.is_available() else "cpu"
# codec_model = EncodecModel.encodec_model_24khz()
# codec_model.set_target_bandwidth(6)
# codec_model.to(device)

text_data = TextData()
protos_path = sys.argv[1] 
sampleR = int(sys.argv[2])
ROOT_DIR = "sample_data"
with open(protos_path, "rb") as f:
    while True:
        d = f.read(4)
        if len(d) != 4:
            break
        length = struct.unpack('I', d)[0]
        print('length', length)
        text_data.ParseFromString(f.read(length))
        #print(text_data)
        

        #p = os.path.join(ROOT_DIR, text_data.source, text_data.name)
        for i in range(len(text_data.sentences)):
            if random.randint(0, sampleR) != 0:
                continue
            #os.makedirs(p, exist_ok=True)
            s = text_data.sentences[i]
            print(s)
            sys.exit()
            '''
            with open(os.path.join(p, str(i) + ".py"), "w") as text_file:
                text_file.write(s.text + "\n" + " ".join(s.phones))
            npy_file_path = os.path.join(p, str(i) + ".npy")
            data = [sem.values for sem in s.semantics]
            with open(npy_file_path, "wb") as npy_file:
                np.save(npy_file, data)

            with torch.no_grad():
                out_wav = codec_model.decode([(torch.tensor(data, device=device).unsqueeze(0), None)])
                torchaudio.save(os.path.join(p, str(i) + ".wav"), out_wav[0].cpu(), codec_model.sample_rate)
        '''




