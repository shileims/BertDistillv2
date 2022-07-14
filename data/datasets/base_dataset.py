from .register import Datasets
from abc import ABC, abstractmethod

@Datasets.register_module
class BaseDataset(ABC):

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass