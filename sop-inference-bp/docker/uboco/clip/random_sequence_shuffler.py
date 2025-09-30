from torch.utils.data.sampler import Sampler
import numpy as np


class RandomSequenceSampler(Sampler):

    def __init__(self, n_sample, seq_len, deterministic=False):
        self.n_sample = n_sample
        self.seq_len = seq_len
        self.deterministic = deterministic
        # Set a fixed seed for deterministic behavior
        if deterministic:
            self.seed = 42
        else:
            self.seed = None

    def _pad_ind(self, ind):
        zeros = np.zeros(self.seq_len - self.n_sample % self.seq_len)
        ind = np.concatenate((ind, zeros))
        return ind

    def __iter__(self):
        idx = np.arange(self.n_sample)
        if self.n_sample % self.seq_len != 0:
            idx = self._pad_ind(idx)
        idx = np.reshape(idx, (-1, self.seq_len))
        
        # Use deterministic shuffling if requested
        if self.deterministic:
            # Save current random state
            current_state = np.random.get_state()
            # Set deterministic seed
            np.random.seed(self.seed)
            np.random.shuffle(idx)
            # Restore previous random state
            np.random.set_state(current_state)
        else:
            np.random.shuffle(idx)
            
        idx = np.reshape(idx, (-1))
        return iter(idx.astype(int))

    def __len__(self):
        return self.n_sample + (self.seq_len - self.n_sample % self.seq_len)
