import torch
import argparse
from src.multiview import load_model, pretrain, finetune, evaluate_classifier
from torch.optim import AdamW
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import wandb
import shutil
from torch.utils.data import DataLoader
from src.downstream_dataset import CustomBIPDataset
from src.ieegdataset import IEEGDataset
import pandas as pd
import datetime
from utils import check_output_path, read_threshold_sub, load_data_bip, create_readme , stratified_split
from rich import print
import numpy as np
import torch
from torch.utils.data import Dataset
import mne  # Library for reading EDF files
import random 
os.environ['WANDB_DISABLED'] = 'true'
def seed_everything(seed_value):
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print("\nSeed is set to:", seed_value)



def main(args):
    args.train_mode = 'pretrain' if args.pretrain and not args.finetune else 'finetune' if args.finetune and not args.pretrain else 'both'
    args.standardize_epochs = 'channelwise'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = read_threshold_sub(args.sub_list)
    print('Number of subjects loaded:', len(filenames))
    if args.pretrain:
        
        
        
        pretrain_dataset = IEEGDataset(
            filenames=filenames,
            sample_keys=['inputs','labels'],
            chunk_len=args.chunk_len,
            overlap=0,#args.ovlp,
            root_path=args.root_path,
            normalization=args.normalization
        )
        
        # Split the dataset into train and validation
        train_size = int(0.8 * len(pretrain_dataset))
        val_size = len(pretrain_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(pretrain_dataset, [train_size, val_size])

        pretrain_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        pretrain_val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        
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
        print('Finetuning model')
        finetune_path = args.finetune_path
        all_data, event_files, subjects = load_data_bip(finetune_path)
        print('Finetuning on', len(all_data), 'subjects')
        
        finetune_dataset = CustomBIPDataset(
            file_paths=all_data,
            labels=event_files,
            chunk_len=args.chunk_len,
            overlap=124,
            normalization=args.normalization,
            standardize_epochs=args.standardize_epochs
        )
        # finetune_dataset = EDFDataset(file_paths, chunk_len=512, overlap=0, normalization=True)
        print('Finetuning on', len(finetune_dataset), 'samples')
        for i, k in enumerate(finetune_dataset):
            print(i)
            print('DATA : ',k[0].shape)
            print("LABEL : ",k[1].shape)
            break
        labels = np.array([label.item() for _, label in finetune_dataset])
        print(labels)
        train_dataset, val_dataset, test_dataset = stratified_split(
            finetune_dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, labels=np.array(labels)
        )

        # DataLoaders
        finetune_loader = DataLoader(train_dataset, batch_size=args.target_batchsize, shuffle=True)
        finetune_val_loader = DataLoader(val_dataset, batch_size=args.target_batchsize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.target_batchsize, shuffle=False)

        # for i, k in enumerate(finetune_loader):
        #     print(i)
        #     print(k[0].shape)
        #     print(k[1].shape)
        #     break
        group = f'{args.pretraining_setup}_{args.loss}'
        wandb.init(project = 'MultiView', group = group, config = args)
        
        if args.load_model:
            pretrained_model_path = f'pretrained_models/MultiView_{args.pretraining_setup}_{args.loss}/pretrained_model.pt'
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        
        num_classes = 2  # binary classification for BIP dataset
        model, _ = load_model(args.pretraining_setup, device, finetune_dataset[0][0].shape[1], finetune_dataset[0][0].shape[0], num_classes, args)
        
        if args.optimize_encoder:
            optimizer = AdamW(model.parameters(), lr=args.ft_learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = AdamW(model.classifier.parameters(), lr=args.ft_learning_rate, weight_decay=args.weight_decay)

        output_path = f'finetuned_models/MultiView_{args.pretraining_setup}_{args.loss}'
        output_path = check_output_path(output_path)
        print('Saving finetuned outputs in', output_path)
        

        finetune(model,
                finetune_loader,
                finetune_val_loader,
                args.finetune_epochs,
                optimizer,
                None,  # No class weights
                device,
                test_loader=test_loader if args.track_test_performance else None,
                early_stopping_criterion=args.early_stopping_criterion,
                backup_path=output_path)

        if args.save_model:
            path = f'{output_path}/finetuned_model.pt'
            torch.save(model.state_dict(), path)

        # Evaluate the finetuned model
        accuracy, prec, rec, f1 = evaluate_classifier(model, test_loader, device)
        wandb.config.update({'Test accuracy': accuracy, 'Test precision': prec, 'Test recall': rec, 'Test f1': f1})
        wandb.finish()


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
    # parser.add_argument('--data_path', type = str, default = 'sleepps18.yml')
    # parser.add_argument('--finetune_path', type = str, default = 'sleepedf.yml')
    # # whether or not to sample balanced during finetuning
    # parser.add_argument('--balanced_sampling', type = str, default = 'finetune')
    # # number of samples to finetune on. Can be list for multiple runs
    # parser.add_argument('--sample_generator', type = eval, nargs = '+', default = [10, 20, None])

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
    # parser.add_argument('--sample_pretrain_subjects', type = eval, default = False)
    # parser.add_argument('--sample_finetune_train_subjects', type = eval, default = False)
    # parser.add_argument('--sample_finetune_val_subjects', type = eval, default = False)
    # parser.add_argument('--sample_test_subjects', type = eval, default = False)

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
    parser.add_argument('--ft_ovlp', type=int, default=128, help='Overlap between chunks')
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
    main(args)
    current_time = datetime.datetime.now()
    print("\nScript ends at: ", current_time.strftime("%H:%M:%S"))
    cleanup()