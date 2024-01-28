from torch.utils.data import Sampler, RandomSampler, SequentialSampler
import numpy as np
import random

class BatchSampler:
    def __init__(self, sampler, batch_size, drop_last, multiscale_step, img_sizes):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes
            
    def __iter__(self):
        num_batch = 0
        batch = []
        size = self.img_sizes[0]
        for idx in self.sampler:
            batch.append([idx, size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch += 1
                batch = []
                if self.multiscale_step and num_batch%self.multiscale_step == 0:
                    size = self.img_sizes[random.randint(0, len(self.img_sizes)-1)]
        
        if len(batch) > 0 and not self.drop_last:
            yield batch
        
    def __len__(self):
        if self.drop_last:
            return len(self.sampler)//self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size