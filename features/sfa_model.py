"""
Train and evaluate an autoencoder

Author(s): Daniela Wiepert
Last modified: 04/19/2025
"""
#IMPORTS
##built-in
import argparse
import json
import os
from pathlib import Path
import warnings
import random
from typing import List, Union, Tuple, Dict
##third-party
import torch
from torch.utils.data import DataLoader
##local
from emaae.models import CNNAutoEncoder
from emaae.loops import train, set_up_train, evaluate
from feature_extraction import VideoDataset, residualPCA
from calflops import calculate_flops


def custom_collatefn(batch) -> torch.tensor:
    """
    Custom collate function to put batch together 

    :param batch: batch from DataLoader object
    :return: collated batch in proper format
    """
    warnings.filterwarnings("ignore")

    feat_list = []
    file_list = []
    max_t = 0
    for b in batch:
        f = torch.transpose(b['features'],0,1)
        if f.shape[-1] > max_t:
            max_t = f.shape[-1]
        feat_list.append(torch.transpose(b['features'],0,1))
        file_list.append(b['files'])

    for i in range(len(feat_list)):
        f = feat_list[i]
        if f.shape[-1] != max_t:
            new_f = torch.nn.functional.pad(f,(0,max_t-f.shape[-1]), mode="constant", value=0)
            feat_list[i] = new_f

    return {'features':torch.stack(feat_list, 0), 'files':file_list}

class DatasetSplitter:
    """
    Split into train/val/test splits based on story

    :param stories: List of str stories
    :param output_dir: str/path object, path to save splits to
    :param num_splits: int, number of splits to generate (default = 5)
    :param train_ratio: float (default = 0.8)
    :param val_ratio: float (default = 0.1)
    """
    def __init__(self, paths: List[Union[str,Path]], train_ratio:float = 0.8, val_ratio:float = 0.1, seed=42):
        self.paths = [str(p) for p in paths]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed=seed

    def split_paths(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Split paths based on train and val ratio

        :return train: list of train stories
        :return val_: list of validation stories
        :return test: list of test stories
        """
        random.seed(self.seed)
        random.shuffle(self.paths)
        num_train = int(len(self.paths) * self.train_ratio)
        num_val = int(len(self.paths) * self.val_ratio)
        
        train = self.paths[:num_train]
        if self.train_ratio + self.val_ratio == 1:
            val = self.paths[num_train:]
            test = self.paths 
        else:
            val= self.paths[num_train:num_train + num_val]
            test = self.paths[num_train + num_val:]
        
        return train, val, test

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='Path to directory with training data.')
    parser.add_argument('--video_dir', type=str, default='',
                        help='Path to directory with training data.')
    parser.add_argument("--train_ratio", type=float, default=.9)
    parser.add_argument("--val_ratio", type=float, default=.1)
    parser.add_argument('--out_dir', type=str, required=True,
                        help="Specify a local directory to save configuration files to. If not saving features to corral, this also specifies local directory to save files to.")
    parser.add_argument("--encode", action="store_true", 
                        help='Save encodings of test features')
    parser.add_argument("--decode", action="store_true", 
                        help='Save decodings of test features')
    ##model specific
    model_args = parser.add_argument_group('model', 'model related arguments')
    model_args.add_argument('--model_config', type=str, default=None, 
                                help='Path to model config json. Default = None')
    model_args.add_argument('--model_type', type=str, default='cnn', 
                                help='Type of autoencoder to initialize.')
    model_args.add_argument('--input_dim', type=int, default=1408,
                                help='Input dimensions')
    model_args.add_argument('--inner_size', type=int, default=1408,
                                help='Size of encoder representations to learn.')
    model_args.add_argument('--n_encoder', type=int, default=5, 
                                help='Number of encoder blocks to use in model.')
    model_args.add_argument('--initial_ekernel', type=int, default=5, 
                                help='Size of initial kernel to encoding block.')
    model_args.add_argument('--n_decoder', type=int, default=3, 
                                help='Number of decoder blocks to use in model.')
    model_args.add_argument('--initial_dkernel', type=int, default=5, 
                                help='Size of initial kernel to decoding block.')
    model_args.add_argument('--batchnorm_first', action='store_true',
                                help='Indicate whether to use batchnorm before or after nonlinearity.')
    model_args.add_argument('--exclude_final_norm', action='store_true',
                                help='Indicate whether to exclude final normalization layer before encoding.')
    model_args.add_argument('--exclude_all_norm', action='store_true',
                                help='Indicate whether to exclude all normalization layers before encoding.')
    model_args.add_argument('--final_tanh', action='store_true',
                                help='Indicate whether to use tanh activation as final part of decoder.')
    model_args.add_argument('--checkpoint', type=str, default=None, 
                                help='Checkpoint name of full path to model checkpoint.')
    model_args.add_argument('--residual', action='store_true',
                                help='add residual connection in training')
    ##train args
    train_args = parser.add_argument_group('train', 'training arguments')
    train_args.add_argument('--eval_only', action='store_true',
                                help='Specify whether to run only evaluation.')
    train_args.add_argument('--early_stop', action='store_true',
                                help='Specify whether to use early stopping.')
    train_args.add_argument('--patience', type=int, default=500,
                                help='Patience for early stopping.')
    train_args.add_argument('--batch_sz', type=int, default=32,
                                help='Batch size for training.')
    train_args.add_argument('--epochs', type=int, default=500,
                                help='Number of epochs to train for.')
    train_args.add_argument('--lr', type=float, default=1e-3,
                                help='Learning rate.')
    train_args.add_argument('--optimizer', type=str, default='adamw',
                                help='Type of optimizer to use for training.')
    train_args.add_argument('--reconstruction_loss', type=str, default='mse',
                                help='Specify base reconstruction loss type.')
    train_args.add_argument('--encoding_loss', type=str, default='tvl2',
                                help='Specify encoding loss type [l1, tvl2, filter].')
    train_args.add_argument('--cutoff_freq', type=float, default=0.2,
                            help='Cutoff frequency for low pass filter training')
    train_args.add_argument('--n_taps', type=int, default=51,
                            help='n_taps for firwin filter')
    train_args.add_argument('--n_filters', type=int, default=20,
                            help='number of filters for evaluation')
    train_args.add_argument('--weight_penalty', action='store_true',
                                help='Specify whether to add a penalty based on model weights.')
    train_args.add_argument('--alpha', type=float, default=0.001,
                                help='Specify loss weights.')
    train_args.add_argument('--update', action='store_true',
                                help='Specify whether to update alpha.')
    train_args.add_argument('--alpha_epochs', type=int, default=15,
                                help='Specify loss weights.')
    train_args.add_argument('--penalty_scheduler', type=str, default='step',
                                help='Specify what penalty scheduler to use.')
    train_args.add_argument('--lr_scheduler', action='store_true', 
                                help='Specify whether to add an lr scheduler')
    train_args.add_argument('--end_lr', type=float, default=0.0001,
                                help='Specify goal end learning rate.')
    train_args.add_argument('--skip_eval', action='store_true')
    

    args = parser.parse_args()

     # CONNECT TO CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Model training on GPU')
    else:
        device = torch.device("cpu")

    # PREP DIRECTORIES
    args.feat_dir = Path(args.feat_dir)

    args.out_dir = Path(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    assert args.feat_dir.exists()

    if args.encoding_loss == 'filter':
        assert args.cutoff_freq is not None and args.n_taps is not None
        filter_loss = True
        args.alpha = 1
    else:
        filter_loss = False
    # PREP VARIABLES
    ## SET ALPHA
    if args.update and not args.early_stop:
            args.alpha_epochs = args.epochs #if you're updating but not early stopping, assumes it is updating for the entire epoch, so alpha epochs and epochs should be equivalent


    cci_features = None
    print('Loading features from local filesystem.')

    #### GET DATA SPLITS
    paths1 = args.feat_dir.rglob('*.npz')
    paths2 = args.feat_dir.rglob('*.npy')
    paths1 = [p for p in paths1]
    paths2 = [p for p in paths2]
    paths = paths1 + paths2
    ds = DatasetSplitter(paths = paths, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    train_files, val_files, test_files = ds.split_paths()

    print(f'Train size: {len(train_files)}')
    print(f'Val size: {len(val_files)}')
    print(f'Test size: {len(test_files)}')

    test_files = train_files + val_files
    # SET UP DATASETS/DATALOADERS
    test_dataset = VideoDataset(video_root=args.video_dir, dataset="", use_dataset=False, use_existing=True, split=test_files, feature_root=args.feat_dir, to_tensor=True)
    feats = test_dataset.features

    train_dataset = VideoDataset(video_root=args.video_dir, dataset="", use_dataset=False, use_existing=True, features=feats, split=train_files, feature_root=args.feat_dir, to_tensor=True)
    val_dataset = VideoDataset(video_root=args.video_dir, dataset="", use_dataset=False, use_existing=True,features=feats, split=val_files, feature_root=args.feat_dir, to_tensor=True)
    

    if not args.eval_only:
        assert not bool(set(train_dataset.files) & set(val_dataset.files)), 'Overlapping files between train and validation set.'
        #assert not bool(set(train_dataset.files) & set(test_dataset.files)), 'Overlapping files between train and test set.'
        #assert not bool(set(test_dataset.files) & set(val_dataset.files)), 'Overlapping files between val and test set.'
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_sz, shuffle=True, num_workers=0, collate_fn=custom_collatefn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collatefn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collatefn)
    flop_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collatefn )

    # LOAD/SAVE MODEL CONFIG
    if args.model_config is not None:
        with open(args.model_config, "rb") as f:
            model_config = json.load(f)
    else:
        model_config = {'model_type':args.model_type, 'inner_size':args.inner_size, 'n_encoder':args.n_encoder, 'initial_ekernel':args.initial_ekernel, 'n_decoder':args.n_decoder, 'initial_dkernel':args.initial_dkernel, 'input_dim':args.input_dim, 'checkpoint':args.checkpoint,
                        'epochs':args.epochs, 'learning_rate':args.lr, 'batch_sz': args.batch_sz, 'optimizer':args.optimizer, 'reconstruction_loss':args.reconstruction_loss, 'encoding_loss':args.encoding_loss, 
                        'penalty_scheduler':args.penalty_scheduler, 'weight_penalty':args.weight_penalty, 'alpha': args.alpha, 'alpha_epochs':args.alpha_epochs, 'update':args.update, 'early_stop':args.early_stop, 
                        'patience':args.patience, 'batchnorm_first':args.batchnorm_first, 'final_tanh': args.final_tanh, 'lr_scheduler': args.lr_scheduler, 'end_lr':args.end_lr, 'cutoff_freq':args.cutoff_freq, 'n_taps':args.n_taps,
                        'residual':args.residual, 'exclude_final_norm':args.exclude_final_norm, 'exclude_all_norm':args.exclude_all_norm, 'n_filters':args.n_filters}

    if args.eval_only:
        args.inner_size = model_config['inner_size']
        args.lr = model_config['learning_rate']
        args.epochs = model_config['epochs']
        args.batch_sz = model_config['batch_sz']
        args.optimizer = model_config['optimizer']
        args.reconstruction_loss = model_config['reconstruction_loss']
        args.encoding_loss = model_config['encoding_loss']
        args.weight_penalty = model_config['weight_penalty']
        args.alpha = model_config['alpha']
        args.update =  model_config['update']
        args.penalty_scheduler =  model_config['penalty_scheduler']
        args.early_stop = model_config['early_stop']
        args.batchnorm_first = model_config['batchnorm_first']
        args.final_tanh = model_config['final_tanh']
        args.lr_scheduler = model_config['lr_scheduler']
        args.end_lr = model_config['end_lr']
        args.n_encoder = model_config['n_encoder']
        args.n_decoder = model_config['n_decoder']
        args.initial_ekernel = model_config['initial_ekernel']
        args.initial_dkernel = model_config['initial_dkernel']
        args.cutoff_freq = model_config['cutoff_freq']
        args.n_taps = model_config['n_taps']
        args.residual = model_config['residual']
        args.exclude_all_norm = model_config['exclude_all_norm']
        args.n_filters = model_config['n_filters']
        ##NTAPS CUTOFF FREQ

    name_str =  f'model_innersz{args.inner_size}_e{args.n_encoder}_iek{args.initial_ekernel}_d{args.n_decoder}_idk{args.initial_dkernel}_lr{args.lr}e{args.epochs}bs{args.batch_sz}_{args.optimizer}_{args.reconstruction_loss}_{args.encoding_loss}'
    if args.encoding_loss == 'filter':
        name_str += f'c{args.cutoff_freq}n{args.n_taps}'
        if args.residual:
            name_str += '_res'
    ### LOSS - CUTOFF FREQ
    if args.alpha is not None:
        name_str += f'_a{args.alpha}'
    if args.weight_penalty:
        name_str += '_weightpenalty'
    if args.update:
        name_str += f'_{args.penalty_scheduler}'
    if args.early_stop:
        name_str += f'_earlystop'
    if args.batchnorm_first:
        name_str += f'_bnf'
    if args.final_tanh:
        name_str += f'_tanh'
    if args.lr_scheduler:
        name_str += f'_explr{args.end_lr}'
    if args.exclude_all_norm:
        name_str += f'_nonorm'
    if args.exclude_final_norm and not args.exclude_all_norm:
        name_str += f'_nofinalnorm'
    save_path = args.out_dir / name_str
    save_path.mkdir(exist_ok=True)

    if args.model_config is None:
        with open(str(save_path/'model_config.json'), 'w') as f:
            json.dump(model_config,f)

    # INITIALIZE MODEL / LOAD CHECKPOINT IF NECESSARY
    if args.model_type=='cnn':
        model = CNNAutoEncoder(input_dim=model_config['input_dim'], n_encoder=model_config['n_encoder'], n_decoder=model_config['n_decoder'], 
                               inner_size=model_config['inner_size'], batchnorm_first=model_config['batchnorm_first'], final_tanh=model_config['final_tanh'],
                                 initial_ekernel=model_config['initial_ekernel'], initial_dkernel=model_config['initial_dkernel'], exclude_final_norm =model_config['exclude_final_norm'], exclude_all_norm=model_config['exclude_all_norm'])
    else:
        raise NotImplementedError(f'{args.model_type} not implemented.')
    
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        #print(checkpoint.keys())
        #print(model.state_dict().keys())
        model.load_state_dict(checkpoint)
    model = model.to(device)

    for i, data in enumerate(flop_loader):
        if i >= 1:
            break
        print(i)
        print(data)
        inputs = data['features'].to(device)
        input_shape = list(inputs.shape)
        flops, macs, params = calculate_flops(model=model, 
                                            input_shape=input_shape,
                                            output_as_string=True,
                                            output_precision=4)
        print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


    if not args.eval_only:
        optim, criterion, scheduler = set_up_train(model=model, device=device, optim_type=args.optimizer, lr=args.lr, loss1_type=args.reconstruction_loss,
                                        loss2_type=args.encoding_loss, alpha=args.alpha, weight_penalty=args.weight_penalty,
                                        penalty_scheduler=args.penalty_scheduler, lr_scheduler=args.lr_scheduler, epochs=args.alpha_epochs, end_lr=args.end_lr)

        model = train(train_loader=train_loader, val_loader=val_loader, model=model, 
                      device=device, optim=optim, criterion=criterion, lr_scheduler=scheduler, save_path=save_path, 
                      epochs=args.epochs, alpha_epochs=args.alpha_epochs, update=args.update, 
                      early_stop=args.early_stop, patience=args.patience,weight_penalty=args.weight_penalty,
                      filter_loss=filter_loss, maxt=max(train_dataset.maxt, val_dataset.maxt),filter_cutoff=args.cutoff_freq, ntaps=args.n_taps, residual=args.residual)
        
        #SAVE FINAL TRAINED MODEL
        mpath = save_path / 'models'
        mpath.mkdir(exist_ok=True)
        torch.save(model.state_dict(), str(mpath / f'{model.get_type()}_final.pth'))
    
    #Evaluate
    if not args.skip_eval:
        print('Saving results to:', save_path)
        metrics = evaluate(test_loader=test_loader, maxt=test_dataset.maxt, model=model, save_path=save_path, device=device, 
                        encode=args.encode, decode=args.decode, n_filters=args.n_filters,ntaps=args.n_taps)
