import os
import random
import struct
from pathlib import Path

import click


def pb_iter(pb_path):
    with open(pb_path, "rb") as f:
        while True:
            d = f.read(4)
            if len(d) != 4:
                break
            length = struct.unpack('I', d)[0]
            data = f.read(length)
            yield data


@click.command()
@click.option(
    "--input-root", type=click.Path(), default="/Users/protos"
)
@click.option("--output-root", type=click.Path(), default="/Users/split_pbs")
@click.option("--split-cnt", type=int, default=1024)
def main(input_root, output_root, split_cnt):
    os.makedirs(output_root, exist_ok=True)
    output_file_path = []
    for i in range(split_cnt):
        output_file_path.append(os.path.join(output_root, f"data_{i:04d}.protos"))
        
    #input_file_path = [str(p.absolute()) for p in list(Path(input_root).glob('*.protos'))]
    input_file_path = [str(p.absolute()) for p in Path(input_root).rglob('*.proto')] + \
                  [str(p.absolute()) for p in Path(input_root).rglob('*.protos')]
    
    output_files = [open(p, "wb") for p in output_file_path]
    for pb_path in input_file_path:
        cnt = 0
        for data in pb_iter(pb_path):
            output_files[cnt % len(output_files)].write(
                struct.pack("I", len(data)) + data
            )
            cnt += 1
    for p in output_files:
        p.close()
    for p in output_file_path:
        data_list = list(pb_iter(p))
        random.shuffle(data_list)
        with open(p, 'wb') as f:
            for data in data_list:
                f.write(struct.pack("I", len(data)) + data)


if __name__ == '__main__':
    main()
