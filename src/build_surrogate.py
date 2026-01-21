'''
build_surrogate.py: Sample code to train surrogate model on 2D (HFS) and 1D (LFS) data.
                    - LFS data either consists of a single 1D run replicated
                      into 2D dimensions (nxn), or n distinct 1D runs concatenated into nxn
                    - Final model will be saved

Title:  TRANSFER LEARNING ON MULTI-DIMENSIONAL DATA: 
        A NOVEL APPROACH TO NEURAL NETWORK-BASED SURROGATE MODELING
        DOI: 10.1615/JMachLearnModelComput.2024057138

This work adapts the encoder-decoder surrogate model found in:
    "Convolutional Dense Encoder-Decoder Networks":
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    "Deep Autoregressive Neural Networks for High-Dimensional Inverse Problems in Groundwater Contaminant Source Identification"
    https://github.com/cics-nd/cnn-inversion
in order to train on simulation data of different dimensions.

author: A.M. Propp
email: propp@stanford.edu
GitHub: apropp
Updated: May 2024
'''

import csv
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.autograd import Variable
from h5py import File
import h5py
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import copy
import time
import argparse

from dense_ed_multidimensional import *
from utils_surrogate import *


def load_data(filename, high_fidelity=True):
    """ Load data from file, applying necessary transformations. """
    with File(filename, 'r') as f:
        transpose_order = (3, 2, 0, 1) if high_fidelity else (0, 1, 3, 2)
        ks = np.transpose(f['ks' if high_fidelity else 'k'], transpose_order) * (10**14)
        ps = np.transpose(f['ps' if high_fidelity else 'p'], transpose_order)
        ss = np.transpose(f['ss' if high_fidelity else 's'], transpose_order)
    return ks, ps, ss

def main(opt):
    path = '/home/groups/dtartako/transfer_learning/sample_data/' # Path to data repository

    # This would need to be modified for a new dataset
    n_HFS_runs_per_file=300
    n_LFS_runs_per_file=100 if opt['LFS_high_freq'] else 1000
    n_test = opt['n_test']

    # Determine how many files we need to open
    n_HFS_files_needed = int(np.ceil((n_test + opt['n_hfs'] + 10) / n_HFS_runs_per_file))
    n_LFS_files_needed = int(np.ceil((n_test + opt['n_lfs'] + 10) / n_LFS_runs_per_file))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load High-Fidelity (2D) Data
    dat = {}
    for i in range(n_HFS_files_needed):  # Assuming filenames are consistent and sequential for simplification
        filename_full = path + f'DATA_2D_v{i+1}.hdf5'
        ks, ps, ss = load_data(filename_full, high_fidelity=True)
        if i == 0:
            dat['ks_hfs'], dat['ps_hfs'], dat['ss_hfs'] = ks, ps, ss
        else:
            dat['ks_hfs'] = np.concatenate((dat['ks_hfs'], ks), axis=0)
            dat['ps_hfs'] = np.concatenate((dat['ps_hfs'], ps), axis=0)
            dat['ss_hfs'] = np.concatenate((dat['ss_hfs'], ss), axis=0)
    
    # Load Low-Fidelity (1D) Data
    for i in range(n_LFS_files_needed):  # Assuming filenames are consistent and sequential for simplification
        file_root = f'data_apr27_test28_extra{i+1}.h5' if opt['LFS_high_freq'] else f'DATA_1D_LF_v{i+1}.h5' 
        filename_full = path + file_root
        ks, ps, ss = load_data(filename_full, high_fidelity=False)
        if i == 0:
            dat['ks_lfs'], dat['ps_lfs'], dat['ss_lfs'] = ks, ps, ss
        else:
            dat['ks_lfs'] = np.concatenate((dat['ks_lfs'], ks), axis=0)
            dat['ps_lfs'] = np.concatenate((dat['ps_lfs'], ps), axis=0)
            dat['ss_lfs'] = np.concatenate((dat['ss_lfs'], ss), axis=0)
    
    #####################################################
    ## Phase 1
    ##  - Build data-loaders using LFS data
    ##  - Build model and modify to build model_phase1
    ##  - Train model_phase1 using LFS data
    #####################################################
    print('Phase 1 commencing...')

    # Build LFS data loaders
    test_loader_lfs, dat = \
        loader_test(data=dat, num_test=n_test, Nxy=(128,128), bs=10, scale='lfs')
    train_loader = \
        loader_train(data=dat, num_training=opt['n_lfs'], Nxy=(128,128), bs=10, scale='lfs', order=0)

    # Build model
    model_orig = DenseED(1, 16, blocks=(7,12,7), growth_rate=40,
                        drop_rate=0, bn_size=8,
                        num_init_features=128, bottleneck=False).to(device)
    
    model_phase1_orig = DenseED_phase1(model_orig,blocks=(7,12,7)).to(device)

    # Train
    model_phase1, rmse_best = model_train(train_loader=train_loader, test_loader=test_loader_lfs, 
                                      reps=opt['reps_phase1'], n_epochs=opt['n_epochs_phase1'], log_interval=1, 
                                      model_orig=model_phase1_orig, 
                                      lr=opt['lr_phase1'], wd=opt['wd_phase1'], factor=opt['factor_phase1'], min_lr=opt['min_lr_phase1'])
    print(f"PHASE1 RMSE BEST = {rmse_best}")

    #####################################################
    ## Phase 2
    ##  - Build data-loaders using HFS data
    ##  - Modify model_phase1 to build model_phase2
    ##  - Train model_phase2 using HFS data
    #####################################################
    print('Phase 2 commencing...')

    # Build HFS data loaders
    test_loader_hfs, dat = \
        loader_test(data=dat, num_test=n_test, Nxy=(128,128), bs=10, scale='hfs')
    train_loader = \
        loader_train(data=dat, num_training=opt['n_hfs'], Nxy=(128,128), bs=10, scale='hfs', order=0)

    # Build model
    model_phase2_orig = DenseED_phase2(model_orig,model_phase1)

    # First freeze all weights to prevent them from being updated
    for param in model_phase2_orig.parameters():
        param.requires_grad = False

    # Then unfreeze weights for the layers we do want to update
    for param in model_phase2_orig.features.decblock2.parameters():
        param.requires_grad = True
    for param in model_phase2_orig.features.up2.parameters():
        param.requires_grad = True

    # Train
    model_phase2, rmse_best = model_train(train_loader=train_loader, test_loader=test_loader_hfs, 
                                        reps=opt['reps_phase2'], n_epochs=opt['n_epochs_phase2'], log_interval=1, 
                                        model_orig=model_phase2_orig, 
                                        lr=opt['lr_phase2'], wd=opt['wd_phase2'], factor=opt['factor_phase2'], min_lr=opt['min_lr_phase2'])
    print(f"PHASE2 RMSE BEST = {rmse_best}")

    #####################################################
    ## Phase 3
    ##  - Modify model_phase2 to build model_phase3
    ##  - Train model_phase3 using HFS data
    #####################################################
    print('Phase 3 commencing...')

    # Build model
    model_phase3_orig = model_phase2

    # Unfreeze all weights
    for param in model_phase3_orig.parameters():
        param.requires_grad = True

    # Train
    model_phase3, rmse_best = model_train(train_loader=train_loader, test_loader=test_loader_hfs, 
                                        reps=opt['reps_phase3'], n_epochs=opt['n_epochs_phase3'], log_interval=1, 
                                        model_orig=model_phase3_orig, 
                                        lr=opt['lr_phase3'], wd=opt['wd_phase3'], factor=opt['factor_phase3'], min_lr=opt['min_lr_phase3'])
    print(f"PHASE3 RMSE BEST = {rmse_best}")

    # Save model
    torch.save(model_phase3.state_dict(), opt['model_filename'])

    # Record results
    results_list = [opt['n_lfs'], opt['n_hfs'], rmse_best, f"{'high' if opt['LFS_high_freq'] else 'low'} freq"]
    with open(opt['results_filename'], 'a') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(results_list)
        f_object.close()


if __name__ == '__main__':
    torch.manual_seed(123) # For reproducible results

    # Define parameters below
    # NOTE: all parameters have default values, so the model can be run as-is
    parser = argparse.ArgumentParser(description='Train a CNN surrogate model for PDE.')
    
    # Data
    parser.add_argument('--LFS_high_freq', default=False, action='store_true', \
                        help='Indicates that low (1D) fidelity data '\
                            'is generated with high-frequency approach')
    parser.add_argument('--n_lfs', type=int, default=100, help='Number of low (1D) fidelity data')
    parser.add_argument('--n_hfs', type=int, default=10, help='Number of high (2D) fidelity data')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test data')
    
    # Saving
    parser.add_argument('--results_filename', type=str, default='results_rep.txt', help='Output filename')
    parser.add_argument('--model_filename', type=str, default='model_rep.pth', help='Surrogate model filename')
    
    # Phase 1 experimental controls
    parser.add_argument('--reps_phase1', type=int, default=2, help='Repetitions for phase 1')
    parser.add_argument('--n_epochs_phase1', type=int, default=100, help='Number of epochs in phase 1')
    parser.add_argument('--lr_phase1', type=float, default=0.0005, help='Learning rate in phase 1')
    parser.add_argument('--wd_phase1', type=float, default=1e-5, help='Weight decay in phase 1')
    parser.add_argument('--factor_phase1', type=float, default=0.6, help='Factor for ReduceLROnPlateau in phase 1')
    parser.add_argument('--min_lr_phase1', type=float, default=1.5e-06, help='Minimum learning rate in phase 1')

    # Phase 2 experimental controls
    parser.add_argument('--reps_phase2', type=int, default=2, help='Repetitions for phase 2')
    parser.add_argument('--n_epochs_phase2', type=int, default=100, help='Number of epochs in phase 2')
    parser.add_argument('--lr_phase2', type=float, default=0.00005, help='Learning rate in phase 2')
    parser.add_argument('--wd_phase2', type=float, default=1e-5, help='Weight decay in phase 2')
    parser.add_argument('--factor_phase2', type=float, default=0.6, help='Factor for ReduceLROnPlateau in phase 2')
    parser.add_argument('--min_lr_phase2', type=float, default=1.5e-06, help='Minimum learning rate in phase 2')

    # Phase 3 experimental controls
    parser.add_argument('--reps_phase3', type=int, default=2, help='Repetitions for phase 3')
    parser.add_argument('--n_epochs_phase3', type=int, default=100, help='Number of epochs in phase 3')
    parser.add_argument('--lr_phase3', type=float, default=0.00001, help='Learning rate in phase 3')
    parser.add_argument('--wd_phase3', type=float, default=1e-5, help='Weight decay in phase 3')
    parser.add_argument('--factor_phase3', type=float, default=0.6, help='Factor for ReduceLROnPlateau in phase 3')
    parser.add_argument('--min_lr_phase3', type=float, default=5e-07, help='Minimum learning rate in phase 3')

    args = parser.parse_args()
    options = vars(args)
    
    main(options)
