import torch
import argparse
from src.multiview import load_model, pretrain, finetune, evaluate_classifier
from torch.optim import AdamW
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
from torch.utils.data.distributed import DistributedSampler
import wandb
import shutil
from torch.utils.data import DataLoader, Dataset
from src.downstream_dataset import CustomBIPDataset
import pandas as pd
import datetime
from utils import check_output_path, read_threshold_sub, load_data_bip, create_readme, stratified_split, stratified_split_with_folds
from rich import print
import random
os.environ['WANDB_DISABLED'] = 'true'
local_rank = int(os.environ["LOCAL_RANK"])
import torch.distributed as dist



def cleanup():
    dist.destroy_process_group()


def seed_everything(seed_value):
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print("\nSeed is set to:", seed_value)

class EDFDataset(Dataset):
    def __init__(self, file_paths, chunk_len=512, overlap=124, normalization=True, segment='train'):
        self.chunk_len = chunk_len
        self.overlap = overlap
        self.normalization = normalization
        self.segment = segment

        self.data = []
        self.labels = []
        self.data_indices = []

        for path in file_paths:
            data = np.load(path)[:10]  # Slice to only use the first 10 channels
            label = 0 if 'class1_erp' in path else 1

            if self.normalization:
                data = self.safe_normalize(data)

            M = data.shape[1]
            total_length = M
            test_start = int(0.95 * total_length) if self.segment == 'test' else 0
            test_end = total_length if self.segment == 'test' else int(0.95 * total_length)
            
            stride = self.chunk_len - self.overlap
            for start in range(test_start, test_end - self.chunk_len + 1, stride):
                self.data.append(data[:, start:start + self.chunk_len])
                self.labels.append(label)

        self.data_indices = list(range(len(self.data)))

    def safe_normalize(self, data):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1
        return (data - mean) / std

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        data_tensor = torch.tensor(data, dtype=torch.float).transpose(0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor.unsqueeze(0)

class NPYIEEGDataset(Dataset):
    def __init__(self, filenames, chunk_len=512, overlap=51, normalization=True):
        self.filenames = filenames
        self.chunk_len = chunk_len
        self.overlap = overlap
        self.normalization = normalization
        self.data_indices = self._create_data_indices()

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        file_idx, start_idx = self.data_indices[index]
        filename = self.filenames[file_idx]
        
        # Load the numpy array and slice to the first 10 channels
        array_data = np.load(filename)[:10]
        
        # Calculate end index ensuring it does not exceed the bounds of the data
        end_idx = min(start_idx + self.chunk_len, array_data.shape[1])
        signal = array_data[:, start_idx:end_idx]

        # Handle cases where the extracted chunk is smaller than expected
        if signal.shape[1] < self.chunk_len:
            padding = np.zeros((signal.shape[0], self.chunk_len - signal.shape[1]))
            signal = np.concatenate((signal, padding), axis=1)

        # Normalize the signal if required
        if self.normalization:
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True)
            std[std == 0] = 1  # avoid division by zero
            signal = (signal - mean) / std

        # Convert the signal to a PyTorch tensor
        signal = torch.tensor(signal, dtype=torch.float32)

        return signal

    def _create_data_indices(self):
        data_indices = []
        for file_idx, filename in enumerate(self.filenames):
            array_data = np.load(filename)[:10]
            total_len = array_data.shape[1]
            stride = self.chunk_len - self.overlap
            num_chunks = (total_len - self.chunk_len) // stride + 1
            for i in range(num_chunks):
                start_idx = i * stride
                data_indices.append((file_idx, start_idx))
        return data_indices


def main(args):
    
    
    args.train_mode = 'pretrain' if args.pretrain and not args.finetune else 'finetune' if args.finetune and not args.pretrain else 'both'
    args.standardize_epochs = 'channelwise'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # filenames = read_threshold_sub(args.sub_list)
    filenames = ['class1_erp_70.npy', 'class4_ersp_70.npy']
    if args.pretrain:
        
        pretrain_dataset = NPYIEEGDataset(
            filenames=filenames,
            chunk_len=512,
            overlap=0,
            normalization=args.normalization
        )
        
        # Split the dataset into train and validation
        train_size = int(0.8 * len(pretrain_dataset))
        val_size = len(pretrain_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(pretrain_dataset, [train_size, val_size])
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)    
        pretrain_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True,sampler=train_sampler)
        pretrain_val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False,sampler=val_sampler)
        
        output_path = f'pretrained_models/MultiView_{args.pretraining_setup}_{args.loss}'
        print('Saving outputs in', output_path)
        create_readme(output_path, [arg for arg in vars(args).items()])
        
        
        # initialize wandb
        wandb.init(project = 'MultiView', group = args.pretraining_setup, config = args)
        
        channels = 0
        time_length = 0
        for i,n in enumerate(pretrain_loader):
            channels = n.shape[0]
            time_length = n.shape[1]
            print(i)
            print(n.shape)
            break

        num_classes = 2  

        model, loss_fn = load_model(args.pretraining_setup, device, channels, time_length, num_classes, args)

        if args.load_model:
            model.load_state_dict(torch.load(output_path, map_location=device))
        
        wandb.config.update({'Pretrain samples': len(pretrain_loader.dataset), 'Pretrain validation samples': len(pretrain_val_loader.dataset)})
        
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        os.makedirs(output_path, exist_ok=True)
        pretrain(model,
                pretrain_loader,
                pretrain_val_loader,
                args.pretrain_epochs,
                optimizer,
                device,
                backup_path=output_path,
                loss_fn=loss_fn,
                log=True)

        model.eval()
        path = f'{output_path}/pretrained_model.pt'
        os.makedirs(output_path, exist_ok=True)
        torch.save(model.state_dict(), path)
        wandb.finish()
    if args.finetune:
        output_path = f'pretrained_models/MultiView_{args.pretraining_setup}_{args.loss}'
        num_classes = 2
        print('Finetuning model')
        
        finetune_file_paths = ['class1_erp_30.npy', 'class4_ersp_30.npy']
        finetune_dataset = EDFDataset(finetune_file_paths, chunk_len=512, overlap=0, normalization=True)
        labels = [sample[1].item() for sample in finetune_dataset]
        print('Finetuning on', len(finetune_dataset), 'samples')

        folds, test_dataset = stratified_split_with_folds(finetune_dataset, labels=np.array(labels), test_ratio=0.1, n_splits=5)
        print('Number of folds:', len(folds))
        print('Number of test samples:', len(test_dataset))
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.target_batchsize, shuffle=False,sampler=test_sampler)

        group = f'{args.pretraining_setup}_{args.loss}'
        wandb.init(project='MultiView', group=group, config=args)

        for fold_idx, (train_dataset, val_dataset) in enumerate(folds):
            print(f'Starting Fold {fold_idx + 1}')
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            train_loader = DataLoader(train_dataset, batch_size=args.target_batchsize, shuffle=True,sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=args.target_batchsize, shuffle=False,sampler=val_sampler)

            model, _ = load_model(args.pretraining_setup, device, finetune_dataset[0][0].shape[1], finetune_dataset[0][0].shape[0], num_classes, args)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
            if args.load_model:
                pretrained_model_path = f'pretrained_models/MultiView_{args.pretraining_setup}_{args.loss}/pretrained_model.pt'
                model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

            if args.optimize_encoder:
                optimizer = AdamW(model.parameters(), lr=args.ft_learning_rate, weight_decay=args.weight_decay)
            else:
                optimizer = AdamW(model.classifier.parameters(), lr=args.ft_learning_rate, weight_decay=args.weight_decay)

            finetune(model,
                    train_loader,
                    val_loader,
                    args.finetune_epochs,
                    optimizer,
                    None,
                    device,
                    test_loader=test_loader if args.track_test_performance and fold_idx == len(folds) - 1 else None,
                    early_stopping_criterion=args.early_stopping_criterion,
                    backup_path=output_path)
        
        if args.track_test_performance:
            accuracy, prec, rec, f1 = evaluate_classifier(model, test_loader, device)
            print(f'Test accuracy: {accuracy}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}')
            wandb.config.update({'Test accuracy': accuracy, 'Test precision': prec, 'Test recall': rec, 'Test f1': f1})

        wandb.finish()

        if args.save_model:
            path = f'{output_path}/finetuned_model.pt'
            if dist.get_rank() == 0:
                # Save or load model
                torch.save(model.state_dict(), model_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
        # training arguments
    parser.add_argument('--local_rank','--local--rank', type=int, default=0)
    
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--job_id', type = str, default = '0')
    parser.add_argument('--seed',type=int, default=0)
    # whether or not to save finetuned models
    parser.add_argument('--save_model', type = eval, default = True)
    parser.add_argument('--load_model', type = eval, default = False)
    parser.add_argument('--pretrain', type = eval, default = True)
    parser.add_argument('--finetune', type = eval, default = False)
    parser.add_argument('--optimize_encoder', type = eval, default = False)
    parser.add_argument('--pretraining_setup', type = str, default = 'MPNN')
    parser.add_argument('--train_mode', type = str)
    # data arguments
    # path to config files. Remember to change paths in config files. 
    parser.add_argument('--data_path', type = str, default = 'sleepps18.yml')
    parser.add_argument('--finetune_path', type = str, default = 'sleepedf.yml')
    # whether or not to sample balanced during finetuning
    parser.add_argument('--balanced_sampling', type = str, default = 'finetune')
    # number of samples to finetune on. Can be list for multiple runs
    parser.add_argument('--sample_generator', type = eval, nargs = '+', default = [10, 20, None])

    # model arguments
    parser.add_argument('--layers', type = int, default = 6)
    # early stopping criterion during finetuning. Can be loss or accuracy (on validation set)
    parser.add_argument('--early_stopping_criterion', type = str, default = None)
    parser.add_argument('--conv_do', type = float, default = 0.1)
    parser.add_argument('--feat_do', type = float, default = 0.1)
    parser.add_argument('--num_message_passing_rounds', type = int, default = 3)
    parser.add_argument('--hidden_channels', type = int, default = 256)
    parser.add_argument('--out_dim', type = int, default = 64)
    parser.add_argument('--embedding_dim', type = int, default = 32)


    # eeg arguments
    # subsample number of subjects. If set to False, use all subjects, else set to integer
    parser.add_argument('--sample_pretrain_subjects', type = eval, default = False)
    parser.add_argument('--sample_finetune_train_subjects', type = eval, default = False)
    parser.add_argument('--sample_finetune_val_subjects', type = eval, default = False)
    parser.add_argument('--sample_test_subjects', type = eval, default = False)

    # optimizer arguments
    parser.add_argument('--loss', type = str, default = 'time_loss', )#ptions = ['time_loss', 'contrastive', 'COCOA'])
    # whether or not to compute performance on test set during training
    parser.add_argument('--track_test_performance', type = eval, default = True)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--ft_learning_rate', type = float, default = 1e-3)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)
    parser.add_argument('--pretrain_epochs', type = int, default = 5)
    parser.add_argument('--finetune_epochs', type = int, default = 1)
    parser.add_argument('--batchsize', type = int, default = 128)
    parser.add_argument('--target_batchsize', type = int, default = 128)

    # Add new arguments for IEEGDataset
    parser.add_argument('--sub_list', type=str, required=False, help='Paths to pretrain data files')
    parser.add_argument('--root_path', type=str, default="", help='Root path for IEEGDataset')

    # Add arguments for CustomBIPDataset
    parser.add_argument('--finetune_data_paths', type=str, nargs='+', required=False, help='Paths to finetune data files')
    parser.add_argument('--chunk_len', type=int, default=512, help='Length of each chunk')
    parser.add_argument('--num_chunks', type=int, default=34, help='Number of chunks')
    parser.add_argument('--ovlp', type=int, default=51, help='Overlap between chunks')
    parser.add_argument('--normalization', type=bool, default=True, help='Whether to normalize the data')

    args = parser.parse_args()
    current_time = datetime.datetime.now()
    seed_everything(args.seed)
    print("\nCurrent Time and Date:")
    print(f"  {current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"\nStarting at  {current_time.strftime('%H:%M:%S')}")
    print("\Input arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    dist.init_process_group("nccl", rank=args.local_rank, world_size=args.world_size)
    main(args)
    current_time = datetime.datetime.now()
    print("\nScript ends at: ", current_time.strftime("%H:%M:%S"))
    cleanup()