import mne
import numpy as np
import os

def save_segments(file_paths, save_dir):
    """
    Load EDF files, segment them into 70% and 30%, and save these segments as .npy files.

    :param file_paths: List of EDF file paths.
    :param save_dir: Directory where the segmented files will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for path in file_paths:
        # Load the EDF file
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()  # Get the data
        # Calculate the indices for 95% and 5% splits
        total_samples = data.shape[1]
        split_idx = int(total_samples * 0.70)

        # Segment the data
        data_95 = data[:, :split_idx]
        data_5 = data[:, split_idx:]

        # Define the base filename without extension
        base_filename = os.path.splitext(os.path.basename(path))[0]

        # Save the 70% data
        np.save(os.path.join(save_dir, f"{base_filename}_70.npy"), data_95)
        # Save the 30% data
        np.save(os.path.join(save_dir, f"{base_filename}_30.npy"), data_5)

        print(f"Saved {base_filename}_70.npy and {base_filename}_30.npy successfully.")

file_paths = ['class1_erp.edf', 'class4_ersp.edf']
save_dir = 'new_data'
save_segments(file_paths, save_dir)
