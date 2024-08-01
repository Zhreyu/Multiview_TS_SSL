import os
import pandas as pd
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch

from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(file_paths, test_size=0.2, finetune_size=0.2):
    """
    Splits data into pretraining, finetuning, and test datasets.
    `test_size` is the proportion of the data to reserve for final testing.
    `finetune_size` is the proportion of the remaining data to use for finetuning.
    """
    # Split off the test dataset first
    remaining_files, test_files = train_test_split(file_paths, test_size=test_size, random_state=42)

    # Split the remaining data into pretraining and finetuning datasets
    pretrain_files, finetune_files = train_test_split(remaining_files, test_size=finetune_size, random_state=42)

    return pretrain_files, finetune_files, test_files


def stratified_split_with_folds(dataset, labels, test_ratio=0.1, n_splits=5):

    train_val_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)), labels, test_size=test_ratio, stratify=labels, random_state=42
    )

    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Data for cross-validation (train/val)
    train_val_labels = labels[train_val_indices]

    # Prepare stratified K-folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in skf.split(train_val_indices, train_val_labels):
        # Convert indices to dataset indices
        train_dataset_indices = [train_val_indices[i] for i in train_idx]
        val_dataset_indices = [train_val_indices[i] for i in val_idx]

        # Create subsets
        train_dataset = torch.utils.data.Subset(dataset, train_dataset_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_dataset_indices)
        folds.append((train_dataset, val_dataset))

    return folds, test_dataset


def stratified_split(dataset, train_ratio, val_ratio, test_ratio, labels):
    # Ensure the sum of ratios equals 1
    print(f"Train ratio: {train_ratio}, Val ratio: {val_ratio}, Test ratio: {test_ratio}")
    print('Sum of ratios:', train_ratio + val_ratio + test_ratio)
    assert round(train_ratio + val_ratio + test_ratio) == 1, "Ratios must sum to 1"

    # First split to separate out the test set
    train_val_indices, test_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=test_ratio,
        stratify=labels,
        random_state=1
    )

    # Adjust the train_ratio to account for the new size of train_val set
    adjusted_train_ratio = train_ratio / (train_ratio + val_ratio)

    # Second split to separate out the train and validation sets
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=1 - adjusted_train_ratio,
        stratify=labels[train_val_indices],
        random_state=1
    )

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def check_output_path(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        i = 1
        while os.path.exists(output_path + f'_v_{i}'):
            i += 1
        output_path = output_path + f'_v_{i}'
        os.makedirs(output_path, exist_ok=True)
    return output_path


def read_threshold_sub(csv_file, lower_bound=2599, upper_bound=1000000):
    df_read = pd.read_csv(csv_file)
    filenames = df_read['filename'].tolist()
    time_lens = df_read['time_len'].tolist()
    filtered_files = []
    for fn, tlen in zip(filenames, time_lens):
        if (tlen > lower_bound) and (tlen < upper_bound):
            filtered_files.append(fn)
    return filtered_files


def get_subject_path_bip(base_dir):
    subject_paths = []
    walk_list = list(os.walk(base_dir))
    sorted_walk_list = sorted(walk_list, key=lambda x: x[0])

    for root, dirs, files in sorted_walk_list:
        if 'FoundationData.npy' in files and 'Events.npy' in files:
            subject_paths.append(root)
    return subject_paths


def load_data_bip(base_dir):
    subject_paths = get_subject_path_bip(base_dir)
    foundation_files = [os.path.join(path, 'FoundationData.npy') for path in subject_paths]
    print(foundation_files)
    print(subject_paths)
    event_files = [os.path.join(path, 'Events.npy') for path in subject_paths]
    #event_files = [os.path.join(path, 'StudyOutcomes.csv')for path in subject_paths]

    subjects = [os.path.basename(path).split('_')[0] for path in subject_paths]
    print(subjects)
    return foundation_files, event_files, subjects


def create_readme(output_dir, args,script_name="readme.txt"):
    os.makedirs(output_dir, exist_ok=True)
    readme_content = f"Script ran: {script_name}\nArguments:\n"
    # Get the absolute path of the current script
    script_path = os.path.abspath(__file__)
    
    # Replace the script name in the path
    base_path = script_path.rsplit('/', 2)[0]
    new_script_path = os.path.join(base_path, script_name)
    
    readme_content += f'Script path: {new_script_path}\n'
    for arg in args:
        readme_content += f"{arg}\n"
    readme_content += f"Date and Time: {datetime.now()}\n"
    readme_path = os.path.join(output_dir, 'readme.txt')
    with open(readme_path, 'w') as readme_file:
        readme_file.write(readme_content)