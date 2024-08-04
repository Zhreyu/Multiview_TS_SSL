import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Generating and saving synthetic data
np.random.seed(42)
num_samples = 1000
num_channels = 20
signal_length = 4000
data = np.random.rand(num_samples, num_channels, signal_length)
labels = np.random.randint(0, 2, size=(num_samples, 1))
np.save('synthetic_data.npy', data)
np.save('synthetic_labels.npy', labels)

# Custom dataset class
class CustomBIPDataset(Dataset):
    def __init__(self, file_paths, labels, chunk_len=512, overlap=124, normalization=True, standardize_epochs=False, bendr_setup=False):
        self.chunk_len = chunk_len
        self.overlap = overlap
        self.standardize_epochs = standardize_epochs
        self.bendr_setup = bendr_setup
        self.normalization = normalization

        self.data = []
        self.labels = []
        self.data_indices = []

        for data_path, label_path in zip(file_paths, labels):
            foundation_data = np.load(data_path)
            events_data = np.load(label_path)

            if np.isnan(foundation_data).any():
                print(f"\nNaN values found in data: {data_path} or {label_path} !!!!!!!")
                continue

            selected_rows = foundation_data[:, :9, :]
            last_row_expanded = np.expand_dims(foundation_data[:, -1, :], axis=1)
            selected_rows = np.concatenate((selected_rows, last_row_expanded), axis=1)
            # selected_rows = foundation_data[np.r_[0:9, 19], :]
            print("Selcted rows shape : ",selected_rows.shape)
            M = selected_rows.shape[0]

            slices = selected_rows[:, :, 3101:4001]
            for i in range(M):
                self.data.append(slices[i])
                self.labels.append(1 if events_data[i, -1] == 1 else 0)

        self.data_indices = self._create_data_indices()

    def __len__(self):
        return len(self.data_indices)

    def _create_data_indices(self):
        data_indices = []
        for file_idx, data in enumerate(self.data):
            total_len = data.shape[1]
            stride = self.chunk_len - self.overlap
            num_chunks = max(1, (total_len - self.chunk_len) // stride + 1)
            for i in range(num_chunks):
                start_idx = i * stride
                data_indices.append((file_idx, start_idx))
        return data_indices

    def __getitem__(self, index):
        file_idx, start_idx = self.data_indices[index]
        data = self.data[file_idx]
        
        # Ensure the chunk does not exceed the bounds of the data
        end_idx = min(start_idx + self.chunk_len, data.shape[1])
        signal = data[:, start_idx:end_idx]

        # Handle cases where the extracted chunk is smaller than expected
        if signal.shape[1] < self.chunk_len:
            # Pad with zeros
            padding = np.zeros((signal.shape[0], self.chunk_len - signal.shape[1]))
            signal = np.concatenate((signal, padding), axis=1)
        
        signal = torch.tensor(signal, dtype=torch.float)
        
        # if self.standardize_epochs:
        #     if self.standardize_epochs == 'total':
        #         signal = (signal - torch.mean(signal)) / torch.std(signal)
        #     elif self.standardize_epochs == 'channelwise':
        #         signal = (signal - torch.mean(signal, dim=1, keepdim=True)) / torch.std(signal, dim=1, keepdim=True)

        label = torch.tensor(self.labels[file_idx], dtype=torch.long).unsqueeze(0)

        return signal, label
    


# Running everything
dataset = CustomBIPDataset(['synthetic_data.npy'], ['synthetic_labels.npy'])
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)


for i, n in enumerate(dataloader):
    x, y = n
    print(f"Batch {i + 1}:")
    print(f"Input shape: {x.shape}, label shape: {y.shape}")
    print(y)
    break

