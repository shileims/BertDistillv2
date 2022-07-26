import os
import torch
import json
import random
import numpy as np
from PIL import Image
from .register import Datasets
from .base_dataset import BaseDataset
from utils import resize as TensorResize
from transformers import RobertaTokenizer, BertTokenizer

@Datasets.register_module
class RetrievalDistill(BaseDataset):
    def __init__(self, data_path, transforms=None, tea_img_size=224, stu_img_size=224, split='coco1k', tokenizer=None):
        assert transforms is not None, f'data augmentation should not be none'
        self.data_path = data_path
        caption_file = os.path.join(self.data_path, 'test_captions.pt')
        self.captions = torch.load(caption_file)
        # FIXME: 5 for coco
        self.num_captions_per_img = 5
        self._tea_img_size = tea_img_size
        self._stu_img_size = stu_img_size

        self.split = split
        if split == 'coco1k':
            eval_img_keys_file = 'test_img_keys_1k.tsv'
        elif split == 'coco5k':
            eval_img_keys_file = 'test_img_keys.tsv'
        elif split == 'flickr':
            eval_img_keys_file = 'test_img_keys.tsv'
        else:
            raise NotImplementedError

        with open(os.path.join(self.data_path, eval_img_keys_file), 'r') as f:
            img_keys = f.readlines()
        self.img_keys = [int(k.strip()) for k in img_keys]
        self.captions = {k: self.captions[k] for k in self.img_keys}
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        else:
            self.tokenizer = tokenizer
        self.transforms = transforms

    def _load_image(self, path):
        # just in case, not sure if there is bad image in scarape IN
        try:
            if '.zip@' in path:
                return self.zipreader.imread(path).convert('RGB')
            else:
                return Image.open(path).convert('RGB')
        except Exception as e:
            print("ERROR IMG LOADED: ", path, e)
            random_img = np.random.rand(self.img_resolution, self.img_resolution, 3) * 255
            img = Image.fromarray(np.uint8(random_img))
            return img.convert('RGB')

    def _clip_pad_1d(self, tensor, max_length):
        # clip when max_length is smaller than current one, to keep the begining idx & end idx
        pad_tensor = tensor.new_zeros((max_length,))
        if tensor.size(0) <= max_length:
            pad_tensor[:tensor.size(0)] = tensor
        else:
            pad_tensor = torch.cat([tensor[:1], tensor[1:max_length-1], tensor[-1:]], dim=0)
        return pad_tensor

    def __getitem__(self, index):
        img_key = self.img_keys[index]
        if 'coco' in self.split:
            image = self._load_image(os.path.join(self.data_path, 'val2014', 'COCO_val2014_{:012}.jpg'.format(img_key)))
        else:
            image = self._load_image(os.path.join(self.data_path, 'test/{}.jpg'.format(img_key)))
        large_img = self.transforms(image)
        if self._tea_img_size != self._stu_img_size:
            small_img = TensorResize(large_img, size=[min(self._tea_img_size, self._stu_img_size), min(self._tea_img_size, self._stu_img_size)])
        else:
            small_img = large_img

        captions = self.captions[img_key]
        if len(captions) > 5:
            # FIXME: ~ 10 samples have 6 captions
            random.shuffle(captions)
            captions = captions[:5]
        if not isinstance(self.tokenizer, RobertaTokenizer) and not isinstance(self.tokenizer, BertTokenizer):
            # used for clip method...
            captions = [self.tokenizer(_text, truncate=True) for _text in captions]
            captions = torch.cat(captions, dim=0)
            return image, captions
        tokenized_caps = []
        for sentence in captions:
            sentence = self.tokenizer(sentence, return_tensors='pt')
            tokenized_caps.append(sentence)
        max_length = 0
        for i, caption in enumerate(tokenized_caps):
            max_length = max(caption['input_ids'].size(1), max_length)
        sentences_input_ids = []
        sentences_attmask = []
        for i, caption in enumerate(tokenized_caps):
            sentences_input_ids.append(self._clip_pad_1d(caption['input_ids'][0], max_length))
            sentences_attmask.append(self._clip_pad_1d(caption['attention_mask'][0], max_length))

        if self._tea_img_size > self._stu_img_size:
            return large_img, small_img, {'input_ids': torch.stack(sentences_input_ids, dim=0), 'attention_mask': torch.stack(sentences_attmask, dim=0)}
        return small_img, large_img, {'input_ids': torch.stack(sentences_input_ids, dim=0), 'attention_mask': torch.stack(sentences_attmask, dim=0)}

    def __len__(self):
        return len(self.img_keys)
