import mne
import numpy as np
import os
import pandas as pd
import torch  # Import torch to handle .pt files

def save_segments(file_paths, save_dir):
    """
    Load EDF files, split data into segments of fixed length, convert to PyTorch tensors,
    save these tensors as .pt files, and generate a CSV with metadata.

    :param file_paths: List of EDF file paths.
    :param save_dir: Directory where the segmented files will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Prepare a list for storing CSV data
    csv_data = []

    for path in file_paths:
        # Load the EDF file
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()  # Get the data
        channels_to_keep = 10  # Only use the first 10 channels
        data = data[:channels_to_keep]  # Reduce to 10 channels
        
        # Segment length
        segment_length = 15360
        # Number of segments
        n_segments = data.shape[1] // segment_length

        # Define the base filename without extension
        base_filename = os.path.splitext(os.path.basename(path))[0]

        # Process each segment
        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = data[:, start_idx:end_idx]

            # Convert numpy array to PyTorch tensor
            tensor_segment = torch.tensor(segment, dtype=torch.float32)

            # Save the tensor segment as a .pt file
            segment_filename = f"{base_filename}_segment_{i}.pt"
            torch.save(tensor_segment, os.path.join(save_dir, segment_filename))

            # Add info to CSV data
            csv_data.append([segment_filename, segment_length])
        
            print(f"Saved {segment_filename} successfully.")

    # Save CSV file with the specified name
    df = pd.DataFrame(csv_data, columns=['filename', 'time_len'])
    df.to_csv(os.path.join(save_dir, 'sub_list.csv'), index=False)

    print("Metadata CSV file 'sub_list.csv' has been saved successfully.")

# Define file paths and save directory
file_paths = ['class1_erp.edf', 'class4_ersp.edf']
save_dir = 'new_data/'

# Call the function to process and save segments
save_segments(file_paths, save_dir)
