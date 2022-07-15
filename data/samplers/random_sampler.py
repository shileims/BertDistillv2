import torch

from .register import Samplers
from .base_sampler import BaseSampler


@Samplers.register_module
class RandomSampler(BaseSampler):
    def __init__(self, data_source, num_samples=-1, generator=None):
        self.data_source = data_source
        self.generator   = generator if generator != None else self.set_generator()
        self.num_samples = num_samples

    def set_generator(self):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def __iter__(self):
        n = len(self.data_source)
        if self.num_samples != -1:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=self.generator).tolist()
            yield from torch.randperm(n, generator=self.generator).tolist()[:self.num_samples % n]
        else:
            yield from torch.randperm(n, generator=self.generator).tolist()

    def __len__(self):
        if self.num_samples == -1:
            return len(self.data_source)
        return self.num_samples


if __name__ == '__main__':
    x = RandomSampler([1,2,3,4,5,6,7,8,9,10], num_samples=5)
    for y in x:
        print(y)
    for y in x:
        print(y)