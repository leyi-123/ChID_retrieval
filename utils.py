import torch
import os
import random
import numpy as np
from transformers import BertTokenizer
import torch.distributed as dist
from copy import deepcopy

class Batcher(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(config.model_type)
        self.cls_id = self.tokenizer.cls_token_id # 101
        self.pad_id = self.tokenizer.pad_token_id # 0
        self.sep_id = self.tokenizer.sep_token_id # 102
        self.mask_id = 103
        self.pad_token_type_id = self.tokenizer.pad_token_type_id

    @staticmethod
    def flatten_list_of_list(list_of_list: list, sep_id: int = None):
        flatten_list = []
        for _list in list_of_list:
            if sep_id is not None:
                flatten_list += _list + [sep_id]
            else:
                flatten_list += _list
        if sep_id is not None:
            assert flatten_list[-1] == sep_id
            flatten_list = flatten_list[:-1]  # delete the final sep_id

        return flatten_list

    @staticmethod
    def truncate(raw_list, max_len, direction):
        assert direction in ['left', 'right']
        if direction == 'left':
            norm_list = raw_list[-max_len:]
        else:
            norm_list = raw_list[:max_len]
        return norm_list

    @staticmethod
    def pad(raw_list, max_len, pad_id, direction):
        assert direction in ['left', 'right'] and max_len >= len(raw_list)
        if direction == 'left':
            norm_list = [pad_id] * (max_len - len(raw_list)) + raw_list
        else:
            norm_list = raw_list + [pad_id] * (max_len - len(raw_list))
        return norm_list

    def get_batch(self, samples):
        candidate_idioms = []
        labels = []
        contents = []
        mask_locations = []

        for item in samples:
            label = item["label"]
            candidate = item["candidate"]
            content = item["content"]
            mask_tag = item["mask_tag"]
            assert content.count(self.mask_id) >= mask_tag + 1
            item_candate_idioms = []
            for idiom in candidate:
                idiom_norm = self.truncate(idiom, max_len=self.config.idiom_max_length, direction='right')
                item_idiom_input_ids = [self.cls_id] + idiom_norm + [self.sep_id]
                item_idiom_input_ids_norm = self.pad(item_idiom_input_ids, max_len=self.config.idiom_max_length + 2,
                                                       pad_id=self.pad_id, direction='right')
                item_candate_idioms.append(item_idiom_input_ids_norm)
            candidate_idioms.append(item_candate_idioms)
            labels.append(label)
            mask_l = location(content, self.mask_id)
            mask_cur = mask_l[mask_tag]
            if self.config.stride: # 取mask位置附近的窗口作为content输入
                if mask_cur >= self.config.stride_length:
                    item_mask_location = self.config.stride_length
                else:
                    item_mask_location = mask_cur

                content_temp = content[mask_cur-self.config.stride_length: mask_cur+self.config.stride_length]
                item_content_norm = [self.cls_id] + content_temp + [self.sep_id]
                item_mask_location += 1
                mask_locations.append(item_mask_location)
                item_content_input_ids_norm = self.pad(item_content_norm, max_len=self.config.content_max_length + 2,
                                                       pad_id=self.pad_id, direction='right')
                contents.append(item_content_input_ids_norm)
            else: # 常规取content的方式
                if len(content) <= self.config.content_max_length:
                    item_content_norm = [self.cls_id] + content + [self.sep_id]
                    item_content_input_ids_norm = self.pad(item_content_norm, max_len=self.config.content_max_length + 2,
                                                         pad_id=self.pad_id, direction='right')
                    contents.append(item_content_input_ids_norm)
                    #mask_locations.append(mask_cur)
                    item_mask_location_l = location(item_content_input_ids_norm, self.mask_id)
                    item_mask_location = item_mask_location_l[mask_tag]
                    mask_locations.append(item_mask_location)
                elif mask_cur <= len(content) // 2:
                    item_content = self.truncate(content, max_len=self.config.content_max_length, direction='right')
                    item_content_norm = [self.cls_id] + item_content + [self.sep_id]
                    item_content_input_ids_norm = self.pad(item_content_norm, max_len=self.config.content_max_length + 2,
                                                         pad_id=self.pad_id, direction='right')
                    contents.append(item_content_input_ids_norm)
                    item_mask_location_l = location(item_content_input_ids_norm, self.mask_id)
                    item_mask_location = item_mask_location_l[mask_tag]
                    mask_locations.append(item_mask_location)
                else:
                    # 假设这种情况只在最后一个mask时出现，且对content后半部分truncate后仍包括最后的mask
                    item_content = self.truncate(content, max_len=self.config.content_max_length, direction='left')
                    item_content_norm = [self.cls_id] + item_content + [self.sep_id]
                    item_content_input_ids_norm = self.pad(item_content_norm, max_len=self.config.content_max_length + 2,
                                                           pad_id=self.pad_id, direction='right')
                    contents.append(item_content_input_ids_norm)
                    item_mask_location_l = location(item_content_input_ids_norm, self.mask_id)
                    item_mask_location = item_mask_location_l[-1]
                    mask_locations.append(item_mask_location)
            #if mask_tag == 0:
            #    temp = content.index(self.mask_id)
            #    if temp <= self.config.content_max_length:
            #        item_content = self.truncate(content, max_len=self.config.content_max_length, direction='right')
            #        item_content_norm = [self.cls_id] + item_content + [self.sep_id]
            #        item_content_input_ids_norm = self.pad(item_content_norm, max_len=self.config.content_max_length + 2,
            #                                             pad_id=self.pad_id, direction='right')
            #        contents.append(item_content_input_ids_norm)
            #        item_mask_location = item_content_input_ids_norm.index(self.mask_id)
            #        mask_locations.append(item_mask_location)
            #    else:
            #        item_content = self.truncate(content, max_len=self.config.content_max_length, direction='left')
            #        item_content_norm = [self.cls_id] + item_content + [self.sep_id]
            #        item_content_input_ids_norm = self.pad(item_content_norm,
            #                                               max_len=self.config.content_max_length + 2,
            #                                               pad_id=self.pad_id, direction='right')
            #        contents.append(item_content_input_ids_norm)
            #        assert self.mask_id in item_content_input_ids_norm
            #        item_mask_location = item_content_input_ids_norm.index(self.mask_id)
            #        mask_locations.append(item_mask_location)
            #else:
            #    temp_content = content
            #    # TODO

        return labels, mask_locations, contents, candidate_idioms
    def get_batch_cross(self, samples): # cross model的输入
        labels = []
        mask_locations = []
        idiom_contents = []

        for item in samples:
            label = item["label"]
            candidate = item["candidate"]
            content = item["content"]
            mask_tag = item["mask_tag"]
            assert content.count(self.mask_id) >= mask_tag + 1
            item_candate_idioms = []
            #for idiom in candidate:
            #    idiom_norm = self.truncate(idiom, max_len=self.config.idiom_max_length, direction='right')
            #    item_idiom_input_ids = [self.cls_id] + idiom_norm + [self.sep_id]
            #    item_idiom_input_ids_norm = self.pad(item_idiom_input_ids, max_len=self.config.idiom_max_length + 2,
            #                                           pad_id=self.pad_id, direction='right')
            #    item_candate_idioms.append(item_idiom_input_ids_norm)
            #candidate_idioms.append(item_candate_idioms)
            labels.append(label)
            mask_l = location(content, self.mask_id)
            mask_cur = mask_l[mask_tag]
            item_mask_location = mask_cur
            if self.config.stride: # 取mask位置附近的窗口作为content输入
                if mask_cur >= self.config.stride_length:
                    item_mask_location = self.config.stride_length
                else:
                    item_mask_location = mask_cur
                content_temp = content[mask_cur - self.config.stride_length: mask_cur + self.config.stride_length]
                item_content_norm = content_temp + [self.sep_id]
                item_content_input_ids_norm = self.pad(item_content_norm, max_len=self.config.content_max_length + 1,
                                                       pad_id=self.pad_id, direction='right')
            else: # 常规取content的方式
                if len(content) <= self.config.content_max_length:
                    item_content_norm = content + [self.sep_id]
                    item_content_input_ids_norm = self.pad(item_content_norm, max_len=self.config.content_max_length + 1,
                                                         pad_id=self.pad_id, direction='right')
                    #contents.append(item_content_input_ids_norm)
                    #mask_locations.append(mask_cur)
                    item_mask_location_l = location(item_content_input_ids_norm, self.mask_id)
                    item_mask_location = item_mask_location_l[mask_tag]
                    #mask_locations.append(item_mask_location)
                elif mask_cur <= len(content) // 2:
                    item_content = self.truncate(content, max_len=self.config.content_max_length, direction='right')
                    item_content_norm = item_content + [self.sep_id]
                    item_content_input_ids_norm = self.pad(item_content_norm, max_len=self.config.content_max_length + 1,
                                                         pad_id=self.pad_id, direction='right')
                    #contents.append(item_content_input_ids_norm)
                    item_mask_location_l = location(item_content_input_ids_norm, self.mask_id)
                    item_mask_location = item_mask_location_l[mask_tag]
                    #mask_locations.append(item_mask_location)
                else:
                    # 假设这种情况只在最后一个mask时出现，且对content后半部分truncate后仍包括最后的mask
                    item_content = self.truncate(content, max_len=self.config.content_max_length, direction='left')
                    item_content_norm = item_content + [self.sep_id]
                    item_content_input_ids_norm = self.pad(item_content_norm, max_len=self.config.content_max_length + 1,
                                                           pad_id=self.pad_id, direction='right')
                    #contents.append(item_content_input_ids_norm)
                    item_mask_location_l = location(item_content_input_ids_norm, self.mask_id)
                    item_mask_location = item_mask_location_l[-1]
                    #mask_locations.append(item_mask_location)
            temp_idiom_contents = []
            for idiom in candidate:
                idiom_norm = self.truncate(idiom, max_len=self.config.idiom_max_length, direction='right')
                item_idiom_input_ids = [self.cls_id] + idiom_norm + [self.sep_id]
                item_idiom_input_ids_norm = self.pad(item_idiom_input_ids, max_len=self.config.idiom_max_length + 2,
                                                       pad_id=self.pad_id, direction='right')
                item_idiom_content = item_idiom_input_ids_norm + item_content_input_ids_norm # cls + idiom + sep + content + sep; length:self.config.idiom_max_length + 2 + self.config.content_max_length + 1
                temp_idiom_contents.append(item_idiom_content)
            idiom_contents.append(temp_idiom_contents)
            item_mask_location += self.config.idiom_max_length + 2
            mask_locations.append(item_mask_location)
        return labels, mask_locations, idiom_contents
def location(l: list, mask: int):
    temp = deepcopy(l)
    num = temp.count(mask)
    #print(num)
    assert num >= 1
    result = list()
    for i in range(num):
        ind = temp.index(mask)
        result.append(ind)
        try:
            temp[ind] = 0 # 把前一个mask消除，这样得到的下一个index就是下一个mask的位置
        except:
            print(temp)
            print(ind)
            exit()
    return result # [mask_num]

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def adapter_to_parallel(model_state: dict):
    if list(model_state.keys())[0].startswith('module.'):
        return model_state
    return {'module.' + k: v for k, v in model_state.items()}

def set_logger(logger, logdir, accelerator=None):
    import logging

    if accelerator is None:
        return logger
    else:
        if accelerator.is_local_main_process:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y/%m/%d %H:%M:%S')

            fh = logging.FileHandler(os.path.join(logdir, f"logging.txt"), mode='w')  # use FileHandler to file
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)

            logger.addHandler(fh)
        else:
            logger.setLevel(logging.WARNING)
            formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y/%m/%d %H:%M:%S')

            fh = logging.FileHandler(os.path.join(logdir, f"logging_subprocess-{dist.get_rank()}.txt"), mode='w')  # use FileHandler to file
            fh.setLevel(logging.WARNING)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def adapter_from_parallel(model_state: dict):
    if not list(model_state.keys())[0].startswith('module.'):
        return model_state
    return {k[7:]: v for k, v in model_state.items()}