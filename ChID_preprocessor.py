import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import re

def create_xinhua(path, path_w):
    print("Begin extracting explanation from xinhua")
    with open(path, encoding="utf-8") as f, open(path_w, "w", encoding="utf-8") as f_w:
        d = {}
        for line in tqdm(f):
            documents = json.loads(line.strip())
            for item in documents:
                idiom = item["word"]
                explanation = item["explanation"]
                d[idiom] = explanation
        t = json.dumps(d, ensure_ascii=False)
        f_w.write(t)
def create_raw(path, path_idiom, path_w):
    print("Begin creating raw data")
    with open(path, encoding="utf-8") as f, open(path_idiom, encoding="utf-8") as f_d, open(path_w, "w", encoding="utf-8") as f_w:
        id = f_d.readline() #
        idiom_dict = json.loads(id.strip()) # dict
        for line in tqdm(f):
            item = json.loads(line.strip())
            groundTruth = item["groundTruth"] # list
            candidates = item["candidates"] # list(list)
            content = item["content"]
            realCount = item["realCount"] # int
            content = content.replace("[MASK]", "")
            content = content.replace("#idiom#", "[MASK]")

            assert len(groundTruth) == realCount
            assert len(candidates) == realCount
            for i in range(realCount):
                groundIdiom = groundTruth[i]
                candidate = candidates[i]
                assert len(candidate) == 7
                label = candidate.index(groundIdiom)
                word_candidate = []
                for idiom in candidate:
                    try:
                        explanation = idiom_dict[idiom]
                    except:
                        explanation = ""
                    word = idiom + " " + explanation
                    word_candidate.append(word)
                raw_item = {"label": label, "candidate": word_candidate, "content": content, "mask_tag": i}
                t = json.dumps(raw_item, ensure_ascii=False)
                f_w.write(f'{t}\n')
def create_tok(path, path_w):
    print("Begin creating tokenize data")
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    with open(path, encoding="utf-8") as f, open(path_w, "w", encoding="utf-8") as f_w:
        for line in tqdm(f):
            item = json.loads(line.strip())
            label = item["label"]
            candidate = item["candidate"]
            content = item["content"]
            mask_tag = item["mask_tag"]
            tok_item = {
                "label" : label,
                "candidate" : [tokenizer(c, add_special_tokens=False)["input_ids"] for c in candidate],
                "content" : tokenizer(content, add_special_tokens=False)['input_ids'],
                "mask_tag" : mask_tag
            }
            f_w.write(f'{json.dumps(tok_item)}\n')

if __name__ == '__main__':
    path_xinhua = "idiom.json"
    path_idiom_dict = "idiom_dict.json"
    path_train = "train_data_1w.json"
    path_train_raw = "train_data_1w_raw.json"
    path_train_tok = "train_data_1w_tok.json"
    create_xinhua(path_xinhua, path_idiom_dict)
    create_raw(path_train, path_idiom_dict, path_train_raw)
    create_tok(path_train_raw, path_train_tok)