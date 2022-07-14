import torch

from .register import Collators
from .base_collator import BaseClollator

@Collators.register_module
class SStockCollatorDistill(BaseClollator):
    def __init__(self, overall_maxlength=48, pad_id=1):
        # in roberta: pad id is 1. for BERT: 0, for T5: 0 for XLNET: 5
        self.overall_maxlength = overall_maxlength
        self.pad_id = pad_id

    def _clip_pad_1d(self, tensor, max_length, pad_id=0):
        # clip when max_length is smaller than current one, to keep the begining idx & end idx
        pad_tensor = tensor.new_zeros((max_length,)) + pad_id
        if tensor.size(0) <= max_length:
            pad_tensor[:tensor.size(0)] = tensor
        else:
            pad_tensor = torch.cat([tensor[:1], tensor[1:max_length-1], tensor[-1:]], dim=0)
        return pad_tensor

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        sentences_input_ids, sentences_attmask, tea_images, stu_images = [], [], [], []
        max_length = 0
        for i, ibatch in enumerate(batch):
            tea_images.append(ibatch[0])
            stu_images.append(ibatch[1])
            max_length = max(ibatch[2]['input_ids'].size(1), max_length)
        max_length = min(self.overall_maxlength, max_length)

        for i, ibatch in enumerate(batch):
            sentences_input_ids.append(self._clip_pad_1d(ibatch[2]['input_ids'][0], max_length, pad_id=self.pad_id))
            sentences_attmask.append(self._clip_pad_1d(ibatch[2]['attention_mask'][0], max_length))
        tea_images = torch.stack(tea_images, dim=0)
        stu_images = torch.stack(stu_images, dim=0)
        return (tea_images, stu_images,
                {'input_ids': torch.stack(sentences_input_ids, dim=0), 'attention_mask': torch.stack(sentences_attmask, dim=0)},)