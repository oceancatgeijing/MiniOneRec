import fire
import pandas as pd
import json
from tqdm import tqdm


def parse_cuda_list(cuda_list):
    if isinstance(cuda_list, int):
        return [cuda_list]
    if isinstance(cuda_list, str):
        return [int(value.strip()) for value in cuda_list.split(',') if value.strip()]
    return [int(value) for value in cuda_list]

def merge(input_path, output_path, cuda_list):
    cuda_list = parse_cuda_list(cuda_list)
    if not cuda_list:
        raise ValueError("cuda_list must contain at least one device")
    data = []
    for i in tqdm(cuda_list):
        with open(f'{input_path}/{i}.json', 'r') as f:
            data.extend(json.load(f))
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(merge)
