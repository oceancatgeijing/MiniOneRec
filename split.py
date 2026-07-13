import fire
import os
import pandas as pd


def parse_cuda_list(cuda_list):
    if isinstance(cuda_list, int):
        return [cuda_list]
    if isinstance(cuda_list, str):
        return [int(value.strip()) for value in cuda_list.split(',') if value.strip()]
    return [int(value) for value in cuda_list]

def split(input_path, output_path, cuda_list):
    cuda_list = parse_cuda_list(cuda_list)
    df = pd.read_csv(input_path)
    # df = df.sample(frac=1).reset_index(drop=True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_len = len(df)
    cuda_num = len(cuda_list)
    if cuda_num == 0:
        raise ValueError("cuda_list must contain at least one device")
    for i in range(cuda_num):
        start = i * df_len // cuda_num
        end = (i+1) * df_len // cuda_num
        df[start:end].to_csv(f'{output_path}/{cuda_list[i]}.csv', index=False)
        
if __name__ == '__main__':
    fire.Fire(split)
