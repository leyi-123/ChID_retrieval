import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from copy import deepcopy

def read_cached_data_tqdm(file_path, debug=False, debug_num=100):
    examples = []
    print('reading cached data ...')
    num_lines = int(str(os.system(f'wc -l {file_path}')).split()[0])
    with open(file_path, 'r', encoding='utf8') as f, tqdm(total=num_lines) as pbar:
        for line in f:  # generator
            line = line.strip()
            example = json.loads(line)
            examples.append(example)
            del line
            pbar.update(1)
            if debug and len(examples) >= debug_num:
                break
    print('num of cached examples : {}'.format(len(examples)))
    return examples

class ChIDDataset(Dataset):
    def __init__(self, data_path, config):
        super(ChIDDataset, self).__init__()
        self.data_path = data_path
        self.config = config
        self.is_training = 'train' in data_path
        self._data = read_cached_data_tqdm(data_path, debug=config.debug, debug_num=config.debug_num)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        temp_item = deepcopy(self._data[idx])  # 拷贝出来一个用来处理数据
        return temp_item
