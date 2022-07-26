import os
import json
import random
import copy
import base64
import io
import numpy as np
import re

from PIL import Image
from .register import Datasets
from .base_dataset import BaseDataset
from transformers import RobertaTokenizer, BertTokenizer
from utils import resize as TensorResize


"""
SStock data == BING data
"""

@Datasets.register_module
class SStockDistill(BaseDataset):
    def __init__(self, data_path, transforms=None, tea_img_size=224, stu_img_size=224, is_train=True, fix_length=1483257, debug=False):
        assert transforms is not None, f'data augmentation should not be none'
        if not debug:
            data_name = {True: '0,1,2,3,10,11,12', False: '1'}
        else:
            data_name = {True: '0', False: '0'}
        self._data_name = data_name
        self._data_path = data_path
        self._tea_img_size = tea_img_size
        self._stu_img_size = stu_img_size
        self.database   = []
        self.fix_length = fix_length
        self.load_data_anno(self._data_name.get(is_train, None))
        self.trans = transforms
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def load_data_anno(self, dataset_name):
        assert dataset_name is not None, f'dataset_name should not be none'
        datasets = dataset_name.strip().split(',')
        full_info = []
        for data_idx in datasets:
            json_path = os.path.join(self._data_path, f'split_{data_idx}.json')
            if not os.path.exists(json_path):
                continue
            print(f'Reading json data from {json_path}')
            _full_info = json.load(
                open(os.path.join(self._data_path, f'split_{data_idx}.json')))
            full_info.extend(list(_full_info.values()))
        full_info = full_info[:self.fix_length]

        if len(full_info) < self.fix_length:
            print(f' warning :: sstock only have {len(full_info)}, need {self.fix_length}, resample to it!!')
            pad_info = []
            for _j in range((self.fix_length - len(full_info))//len(full_info)):
                full_info_shuffle = copy.deepcopy(full_info)
                random.shuffle(full_info_shuffle)
                pad_info += full_info_shuffle

            full_info_shuffle = copy.deepcopy(full_info)
            random.shuffle(full_info_shuffle)
            pad_info += full_info_shuffle[:(self.fix_length - len(full_info))%len(full_info)]

            full_info += pad_info

        for info in full_info:
            self.database.append([info['img_caption'], os.path.join(self._data_path,
                                                                    f'split_{info["img_location"]}.tsv/{info["lineidx_ptr"]}')])

    def deconfusing(self, string):
        confuse_head = r'this is a head'.encode('utf-8')
        if string.startswith(confuse_head):
            confuse_code = b'\xff\xdb\x00C\x00\x02\x01'
            string = string[len(confuse_head):]
            result = re.search(b'\xff\xda', string)
            startofscan = result.span()[0]
            return string[:startofscan - len(confuse_code)] + string[startofscan:]
        else:
            # no confusing for Laion dataset
            return string

    def _load_image(self, path):
        assert '.tsv/' in path, f'tsv not in {path}'
        try:
            tsv_name, lineidx = path.split('.tsv/')
            _fp = open(tsv_name + '.tsv', 'r')
            _fp.seek(int(lineidx))
            _, img = [s.strip() for s in _fp.readline().split('\t')]
            img = base64.b64decode(img)
            img = self.deconfusing(img)
            img = Image.open(io.BytesIO(img))
            _fp.close()
            img = img.convert("RGB")
            self.last_img = img
            return img, True
        except Exception as e:
            print("ERROR IMG (.tsv) LOADED: ", path, e)
            return None, False

    def __getitem__(self, item):
        idb = self.database[item]
        # images
        raw_img, success_loaded = self._load_image(idb[1])
        large_img = self.trans(raw_img)
        if self._tea_img_size != self._stu_img_size:
            small_img = TensorResize(large_img, size=[min(self._tea_img_size, self._stu_img_size), min(self._tea_img_size, self._stu_img_size)])
        else:
            small_img = large_img

        # texts
        sentence = idb[0]
        sentence_features = self.tokenizer(sentence, return_tensors='pt')
        if self._tea_img_size > self._stu_img_size:
            return large_img, small_img, sentence_features
        return small_img, large_img, sentence_features

    def __len__(self):
        if self.fix_length != -1:
            return self.fix_length
        return len(self.database)
