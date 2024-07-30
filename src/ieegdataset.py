import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, sosfiltfilt

import torch
from torch.utils.data import Dataset
import numpy as np
import os

class IEEGDataset(Dataset):
    def __init__(self, filenames, sample_keys, chunk_len=512, overlap=51, root_path="", population_mean=0, population_std=1, normalization=True):
        if root_path == "":
            self.filenames = filenames
        else:
            self.filenames = [root_path + fn for fn in filenames if os.path.isfile(root_path+fn)]
            self.root_path = root_path

        print("\nNumber of subjects loaded: ", len(self.filenames))
        
        self.chunk_len = chunk_len  # 2 seconds at 256Hz
        self.overlap = overlap      # 0.2-second overlap
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization

        self.data_indices = self._create_data_indices()

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        file_idx, start_idx = self.data_indices[index]
        filename = self.filenames[file_idx]
        
        # Load the entire tensor
        tensor_data = self.load_tensor(filename)
        
        # Ensure the chunk does not exceed the bounds of the data
        end_idx = min(start_idx + self.chunk_len, tensor_data.shape[1])
        signal = tensor_data[:, start_idx:end_idx]

        # Handle cases where the extracted chunk is smaller than expected
        if signal.shape[1] < self.chunk_len:
            # Pad with zeros
            padding = np.zeros((signal.shape[0], self.chunk_len - signal.shape[1]))
            signal = np.concatenate((signal, padding), axis=1)
        
        if self.do_normalization:
            signal = (signal - self.mean) / self.std
        
        signal = torch.tensor(signal).float()
        
        # channel_data = (signal-torch.mean(signal, axis = 1)[:,np.newaxis])/torch.std(signal, axis = 1)[:,np.newaxis]
        # channel_data = signal 
        # print('cd shape: ',channel_data.shape)
        return signal
    def load_tensor(self, filename):
        tensor_data = torch.load(filename)
        return tensor_data.numpy()

    def _create_data_indices(self):
        data_indices = []
        for file_idx, filename in enumerate(self.filenames):
            tensor_data = self.load_tensor(filename)
            
            total_len = tensor_data.shape[1]
            stride = self.chunk_len - self.overlap
            num_chunks = max(1, (total_len - self.chunk_len) // stride + 1)
            for i in range(num_chunks):
                start_idx = i * stride
                data_indices.append((file_idx, start_idx))
        return data_indices
    
