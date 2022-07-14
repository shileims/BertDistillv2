from abc import ABC, abstractmethod
from .register import Collators

@Collators.register_module
class BaseClollator(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass