import os.path
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import sys
import click
import numpy as np
from loguru import logger
from tqdm import tqdm

from protos.text_data_pb2 import Semantics, Sentence, TextData
from protos.text_data_stream import pack_pb_stream
import random


def load_filelist(path):
    """
    Load a Bert-VITS2 style filelist.
    """

    files = set()
    results = []
    count_duplicated, count_not_found = 0, 0

    LANGUAGE_TO_LANGUAGES = {
        "zh": ["zh", "en"],
        "jp": ["jp", "en"],
        "en": ["en"],
        "de": ["de"],
        "fr": ["fr", "en"],  
        "ko": ["ko", "en"]
    }

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            splits = line.strip().split("|")
            #print('splits', splits)
            if len(splits)==5:
                filename, speaker, language, text, phones = splits
            elif len(splits)==4:
                filename, speaker, language, text = splits
                phones = None
            else:
                logger.warning(f"Invalid line: line_idx:{idx}, line_data:{line, splits}")
                continue
            file = Path(filename)
            language = language.strip().lower()

            if language == "ja":
                language = "jp"

            assert language in ["zh", "jp", "en", "fr", "de", "ko"], f"Invalid language {language}"
            languages = LANGUAGE_TO_LANGUAGES[language]

            if file in files:
                logger.warning(f"Duplicated file: {file}")
                count_duplicated += 1
                continue

            results.append((file, speaker, languages, text, phones))

    if count_duplicated > 0:
        logger.warning(f"Total duplicated files: {count_duplicated}")

    if count_not_found > 0:
        logger.warning(f"Total files not found: {count_not_found}")

    return results


def task_generator_filelist(filelist, wav_root, vq_root, spk_root):
    grouped_files = defaultdict(list)
    for filename, speaker, languages, text, phones in load_filelist(filelist):
        grouped_files[speaker].append((Path(filename), text, languages, phones))

    logger.info(f"Found {len(grouped_files)} groups in {filelist}")
    items = list(grouped_files.items())
    random.shuffle(items)
    '''
    values:
    [
    (PosixPath('/mnt/nas1/zhangying/data/data24k/english_wavs/8713_300047/8713_300047_000002_000000.wav'), 'His life was passed like this', ['zh', 'en'], None),
    (PosixPath(wav_path), text, languages, phones), 
    ...
    ]
    
    '''
    for speaker, values in items:
        #print('file :', values)
        #sys.exit()
        yield speaker, values, Path(filelist).with_suffix("").name, languages, wav_root, vq_root, spk_root


def run_task_kx_data(task):
    #print('run_task_kx_data task', len(task), task)
    name, subset, source, languages, wav_root, vq_root, spk_root = task

    # Parse the files
    sentences = []
    for file, text, languages, phones in subset:
        #print('vq_root', vq_root, 'file', file, 'wav_root', wav_root)
        np_file = Path(os.path.join(vq_root, os.path.relpath(file, wav_root))).with_suffix(".npy")
        #print('vq np_file', np_file)
        emb_path = Path(os.path.join(spk_root, os.path.relpath(file, wav_root))).with_suffix(".npy")
        #Path(str(np_file).replace('base_semantic', 'base_emb'))
        #print('emb_path', emb_path)
        #sys.exit()
        if np_file.exists() is False:
            logger.warning(f"Can't find token {np_file}")
            continue
        elif emb_path.exists() is False:
            logger.warning(f"Can't find emb {emb_path}")
            continue
        try:
            phones = phones.split(" ")
            code = np.load(np_file).reshape(-1) # [T]
            emb = np.load(emb_path).reshape(-1) # (192,)
            
            #code = data['code'].tolist()
            #emb = data['emb'].tolist()
        except Exception as e:
            logger.error(f"Failed to parse {file}: {e}")
            continue

        sentences.append(
            Sentence(
                text=text,
                phones=phones,
                semantics=code,
                file_path=str(file),
                emb=emb
            )
        )

    if len(sentences) == 0:
        return None
    # Pack the sentences
    return pack_pb_stream(
        TextData(
            source=source,
            name=name,
            languages=languages,
            sentences=sentences,
        )
    )

def run_task(task):
    #name, subset, source, languages, wav_root, vq_root = task
    #name, subset, source, languages, wav_root, vq_root = task
    name, subset, source, languages, wav_root, vq_root, spk_root = task

    # Parse the files
    sentences = []
    for file, text, languages, phones in subset:
        np_file = Path(os.path.join(vq_root, os.path.relpath(file, wav_root))).with_suffix(".npz")
        #print('vq_root', vq_root, 'wav_root', wav_root, 'file', file)
        #print('np_file', np_file)
        
        if np_file.exists() is False:
            logger.warning(f"Can't find {np_file}")
            continue

        try:
            if phones is not None:
                phones = phones.split(" ")
            data = np.load(np_file)
            code = data['code'].tolist()
            emb = data['emb'].tolist()
        except Exception as e:
            logger.error(f"Failed to parse {file}: {e}")
            continue
        # file_path: wav_path
        sentences.append(
            Sentence(
                text=text,
                phones=phones,
                semantics=code,
                file_path=str(file),
                emb=emb
            )
        )

    if len(sentences) == 0:
        return None
    # Pack the sentences
    return pack_pb_stream(
        TextData(
            source=source,
            name=name,
            languages=languages,
            sentences=sentences,
        )
    )



@click.command()
@click.option("--output", type=click.Path(), default="data.protos")
@click.option("--filelist", type=click.Path(), default=None)
@click.option("--num-workers", type=int, default=64)
@click.option("--wav-root", default="wav_root", type=Path)
@click.option("--vq-root", default="vq_root", type=Path)
@click.option("--spk-root", default="spk_root", type=Path)
def main(output, filelist, num_workers, wav_root, vq_root, spk_root):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    dataset_fp = open(output, "wb")
    generator_fn = (
        task_generator_filelist(filelist, wav_root, vq_root, spk_root)
    )

    with Pool(num_workers) as p:
        #for result in tqdm(p.imap_unordered(run_task, generator_fn)):
        
        # codes 和emb 分成两个npy 存储
        for result in tqdm(p.imap_unordered(run_task_kx_data, generator_fn)):
            if result is not None:
                dataset_fp.write(result)

    dataset_fp.close()


if __name__ == "__main__":
    main()

    '''
    protos的内容
    source: file_list的名字
    name: speaker
    languages: "zh"
    languages: "en"
    sentences {
          text: "xxx",
          semantics: codes,
          file_path: wav_path,
          emb: 192_values}
  
    '''
