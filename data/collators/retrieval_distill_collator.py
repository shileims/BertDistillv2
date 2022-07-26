import torch

from .register import Collators
from .base_collator import BaseClollator

@Collators.register_module
class RetrievalCollatorDistill(BaseClollator):
    def __init__(self):
        pass

    def _clip_pad_2d(self, tensor, max_length):
        # clip when max_length is smaller than current one, to keep the begining idx & end idx
        pad_tensor = tensor.new_zeros((tensor.size(0), max_length,))
        pad_tensor[:, :tensor.size(1)] = tensor
        return pad_tensor

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        tea_images = []
        stu_images = []
        sentences_input_ids = []
        sentences_attmask = []
        max_length = 0
        for i, ibatch in enumerate(batch):
            tea_images.append(ibatch[0])
            stu_images.append(ibatch[1])
            max_length = max(ibatch[2]['input_ids'].size(1), max_length)

        for i, ibatch in enumerate(batch):
            sentences_input_ids.append(self._clip_pad_2d(ibatch[2]['input_ids'], max_length))
            sentences_attmask.append(self._clip_pad_2d(ibatch[2]['attention_mask'], max_length))
        tea_images = torch.stack(tea_images, dim=0)
        stu_images = torch.stack(stu_images, dim=0)
        return (tea_images, stu_images,
                {'input_ids': torch.cat(sentences_input_ids, dim=0), 'attention_mask': torch.cat(sentences_attmask, dim=0)},)
