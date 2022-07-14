from .register import Samplers
from abc import ABC, abstractmethod

@Samplers.register_module
class BaseSampler(ABC):

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        pass

    def update_indices(self, new_indices):
        self.img_indices = new_indices

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

