#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;
set -ex
stage=4
stop_stage=4

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
config_path=$1
model_name=$2



# protos
# cp protos/ CosyVoice -r
# pb_opener : cosyvoice/dataset/processor.py:from protos.text_data_pb2 import TextData

#python. mytool/sample_pb_data.py xx.protos 1000

## debug: lsof -i :1234

# hift train
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  
  config_path=conf/cosyvoice2_25hz_hifi2.yaml

  model_name=hiftgan_f0debug
  #25hz_nonstream_5e4_warm 
  train_data=split_protos_v2
  cv_data=val_data/

  
  for model in hifigan; do
    torchrun --nnodes=1 --master_port=30000 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:2345" \
      cosyvoice/bin/train2.py \
      --train_engine $train_engine \
      --config $config_path \
      --train_data $train_data \
      --cv_data $cv_data  \
      --model $model \
      --model_dir `pwd`/exp/hiftgan_24k/$model_name/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/hiftgan_24k/$model_name/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} 
      #\
      #--checkpoint `pwd`/exp/hiftgan_24k/hiftgan/torch_ddp/epoch_47_whole.pt
      
      
  done
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  
  config_path=conf/cosyvoice2_25hz_hifi2_1e4_chunk10.yaml

  model_name=25hz_nonstream_ft_for_stream_sftdata_1e4_v1

  train_data=split_protos_v2

  cv_data=val_data/

  
  for model in flow; do
    torchrun --nnodes=1 --master_port=12345 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train2.py \
      --train_engine $train_engine \
      --config $config_path \
      --train_data $train_data \
      --cv_data $cv_data  \
      --model $model \
      --model_dir `pwd`/exp/cosyvoice_25hz/$model_name/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice_25hz/$model_name/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --checkpoint `pwd`/exp/cosyvoice_25hz/25hz_nonstream_ft_for_stream/torch_ddp/epoch_3_step_1300000.pt

      
  done
fi






# average model
average_num=5
if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 200 ] && [ ${stop_stage} -ge 200 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi
