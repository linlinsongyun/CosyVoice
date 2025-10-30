set -ex
stage=2
stop_stage=3

export PYTHONPATH=CosyVoice:$PYTHONPATH


# Emilia.scp: wav_path|utt_id|lang|utt_id|dur|text
# with_phns.txt : wav_path|utt_id|lang|text|phns
data_phn_lists=(cosyvoice/test_with_phns.txt)




vq_root=VQ_ROOT/

protos_output=test_packed/protos

split_protos_output=test_packed/split_protos

spk_root=/mnt/nas1/kaixuan/data/valle/base_emb

mkdir -p $protos_output
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    api_g2p.py --filelist --output
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for data in "${data_phn_lists[@]}"; do
        #python mytool/extract_s3.py --filelist $data --vq-root test_data/vq_data
        data_tag=$(basename "$data" "_with_phs.txt")
        python mytool/extract_semantic.py --filelist $data --vq-root $vq_root --emb_root $spk_root --wav-root wav_root
    done 
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for data in "${data_phn_lists[@]}"; do
        #python mytool/extract_s3.py --filelist $data --vq-root test_data/vq_data
        data_tag=$(basename "$data" "_with_phs.txt")
        python mytool/build_dataset.py --filelist $data --vq-root $vq_root --spk-root $spk_root --wav-root /mnt/nas1/ --output $protos_output/${data_tag}.proto
    done 
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python mytool/split_pb_file.py --input-root $protos_output --output-root $split_protos_output #--split-cnt 1024
fi



