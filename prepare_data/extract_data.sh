set -ex
stage=2
stop_stage=3

data_phn_lists=(xx.txt)


# Emilia.scp: wav_path|utt_id|lang|utt_id|dur|text
# with_phns.txt : wav_path|utt_id|lang|text|phns

vq_root=xx


protos_output=all_protos/

split_protos_output=split_protos/
spk_root=xx

mkdir -p $protos_output

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for data in "${data_phn_lists[@]}"; do
        #python mytool/extract_s3.py --filelist $data --vq-root test_data/vq_data
        data_tag=$(basename "$data" "_with_phs.txt")
        python mytool/extract_semantic.py --filelist $data --vq-root $vq_root --emb_root $spk_root --wav-root /mnt/nas1/
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for data in "${data_phn_lists[@]}"; do
        #python mytool/extract_s3.py --filelist $data --vq-root test_data/vq_data
        data_tag=$(basename "$data" "_with_phs.txt")
        python mytool/build_dataset.py --filelist $data --vq-root $vq_root --spk-root $spk_root --wav-root /mnt/nas1/ --output $protos_output/${data_tag}.proto
    done
fi
