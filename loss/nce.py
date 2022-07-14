import torch
from .register import Losses
from timm.loss import LabelSmoothingCrossEntropy
from utils import get_rank

@Losses.register_module
class NCE(torch.nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)
            borrow from https://github.com/bl0/moco/blob/master/moco/NCE/NCECriterion.py
            for both i2t and t2i
        """

    def __init__(self, label_smooth=0.1):
        super(NCE, self).__init__()
        if label_smooth > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smooth)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, logits_image, logits_text, label_img=None, label_text=None):
        if label_img is None or label_text is None:
            batch_size_this = logits_image.size(0)
            label_img = label_text = torch.zeros([batch_size_this]).long().to(logits_text.device)
        return (self.criterion(logits_image, label_img) + self.criterion(logits_text, label_text)) / 2

@Losses.register_module
class DistributedNCE(torch.nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)
        borrow from https://github.com/bl0/moco/blob/master/moco/NCE/NCECriterion.py
        for both i2t and t2i
    """
    def __init__(self, label_smooth = 0.1):
        super(DistributedNCE, self).__init__()
        if label_smooth > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smooth)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text, label_img=None, label_text=None):
        if label_img is None or label_text is None:
            batch_size_this = logits_per_image.size(0)
            gpu_index = int(get_rank())
            label_img = label_text = torch.arange(gpu_index*batch_size_this, (gpu_index+1)*batch_size_this).long().to(logits_per_text.device)
        return (self.criterion(logits_per_image, label_img) + self.criterion(logits_per_text, label_text)) / 2