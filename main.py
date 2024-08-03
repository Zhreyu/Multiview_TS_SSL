import torch
import argparse
from src.multiview import load_model, pretrain, finetune, evaluate_classifier
from torch.optim import AdamW
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import wandb
from torch.utils.data import DataLoader
from src.downstream_dataset import CustomBIPDataset
from src.ieegdataset import IEEGDataset
import datetime
from utils import check_output_path, read_threshold_sub, load_data_bip, create_readme  
from rich import print
import numpy as np
import torch
import random
import csv
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
    
    if args.pretrain:
        filenames = read_threshold_sub(args.sub_list)
        # print(filenames)
        print('Number of subjects loaded:', len(filenames))
                
        pretrain_dataset = IEEGDataset(
            filenames=filenames,
            sample_keys=['inputs','labels'],
            chunk_len=args.chunk_len,
            overlap=args.ovlp,
            root_path=args.root_path,
            normalization=args.normalization
        )
        
        # Split the dataset into train and validation
        train_size = int(0.8 * len(pretrain_dataset))
        val_size = len(pretrain_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(pretrain_dataset, [train_size, val_size])

        pretrain_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True,pin_memory=True,num_workers=args.num_workers)
        pretrain_val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False,pin_memory=True,num_workers=args.num_workers)
        
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
        wandb.config.update({'Pretrain samples': len(pretrain_loader.dataset), 'Pretrain validation samples': len(pretrain_val_loader.dataset)})
        
        model, loss_fn = load_model(args.pretraining_setup, device, channels, time_length, num_classes, args)

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
        print('Pretraining done. Model saved in', path)
    if args.finetune:
        print('Finetuning model')
        all_data, event_files, subjects = load_data_bip(args.finetune_data_paths)
        print('Finetuning on', len(all_data), 'subjects')

        # List to store metrics
        metrics_list = []
        for i in range(len(all_data)):
            print(f'Fold {i+1}/{len(all_data)}')
            train_files = all_data[:i] + all_data[i+1:]
            val_files = [all_data[i]]
            train_events = event_files[:i] + event_files[i+1:]
            val_events = [event_files[i]]
            print("Valdiation Subject Path: ",val_files)
            train_dataset = CustomBIPDataset(
                file_paths=train_files,
                labels=train_events,
                chunk_len=args.chunk_len,
                overlap=args.ft_ovlp,
                normalization=args.normalization,
                standardize_epochs=args.standardize_epochs
            )
            val_dataset = CustomBIPDataset(
                file_paths=val_files,
                labels=val_events,
                chunk_len=args.chunk_len,
                overlap=args.ft_ovlp,
                normalization=args.normalization,
                standardize_epochs=args.standardize_epochs
            )
            for i,n in enumerate(train_dataset):
                channels = n.shape[0]
                time_length = n.shape[1]
                break
            
            model, loss_fn = load_model(args.pretraining_setup, device, channels, time_length, num_classes, args)

            if args.load_model:
                model_path = args.pretrained_model_path
                model.load_state_dict(torch.load(model_path, map_location=device))

            train_loader = DataLoader(train_dataset, batch_size=args.target_batchsize, shuffle=True, pin_memory=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=args.target_batchsize, shuffle=False, pin_memory=True, num_workers=args.num_workers)

            group = f'{args.pretraining_setup}_{args.loss}'
            wandb.init(project='MultiView', group=group, config=args)
            
            optimizer = AdamW(model.parameters(), lr=args.ft_learning_rate, weight_decay=args.weight_decay) if args.optimize_encoder else AdamW(model.classifier.parameters(), lr=args.ft_learning_rate, weight_decay=args.weight_decay)

            output_path = check_output_path(f'finetuned_models/MultiView_{args.pretraining_setup}_{args.loss}')
            print('Saving finetuned outputs in', output_path)
            
            finetune(model, train_loader, val_loader, args.finetune_epochs, optimizer, None, device, early_stopping_criterion=args.early_stopping_criterion, backup_path=output_path)

            if args.save_model:
                save_path = f'{output_path}/finetuned_model.pt'
                torch.save(model.state_dict(), save_path)

            # Evaluate the finetuned model
            accuracy, prec, rec, f1 = evaluate_classifier(model, val_loader, device)
            wandb.config.update({'Test accuracy': accuracy, 'Test precision': prec, 'Test recall': rec, 'Test f1': f1, })
            wandb.finish()
            
            # Append metrics to the list
            metrics_list.append({
                'Subject': subjects[i],
                'Accuracy': accuracy,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                })
        
        # Save the metrics to a CSV file
        metrics_file = os.path.join(output_path, 'subject_wise_metrics.csv')
        with open(metrics_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Subject', 'Accuracy', 'Precision', 'Recall', 'F1'])
            writer.writeheader()
            for metrics in metrics_list:
                writer.writerow(metrics)
        
        print(f'Subject-wise metrics saved to {metrics_file}')



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
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--pretrained_model_path', type = str, default = 'pretrained_models/MultiView_MPNN_time_loss/pretrained_model.pt')
    # Add new arguments for IEEGDataset
    parser.add_argument('--sub_list', type=str, required=False, help='Paths to pretrain data files')
    parser.add_argument('--root_path', type=str, default="", help='Root path for IEEGDataset')

    # Add arguments for CustomBIPDataset
    parser.add_argument('--finetune_data_paths', type=str, required=False, help='Paths to finetune data files')
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