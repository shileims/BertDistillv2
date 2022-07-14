import torch

from .register import Samplers
from .base_sampler import BaseSampler


@Samplers.register_module
class SequentialSampler(BaseSampler):
    def __init__(self, data_source, num_samples=-1):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.num_samples != -1:
            for i in range(self.num_samples // n):
                yield from [i*n+j for j in range(n)]
            yield from [(self.num_samples//n)*n+j for j in range(n)][:self.num_samples % n]
        else:
            yield from [j for j in range(n)]

    def __len__(self):
        if self.num_samples == -1:
            return len(self.data_source)
        return self.num_samples


if __name__ == '__main__':
    x = SequentialSampler([1,2,3,4,5,6,7,8,9,10], num_samples=5)
    for y in x:
        print(y)