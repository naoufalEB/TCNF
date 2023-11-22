# Copyright (c) 2019-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import time

file_path = 'C:/Users/naouf/Documents/0-These/2-Scripts/10-Sto_norm_flow/4-Time-changed_NF_2'
os.chdir(file_path)

import lib.utils as utils
import numpy as np
import torch

from bm_sequential import get_test_dataset as get_dataset
from ctfp_tools import build_augmented_model_tabular
from ctfp_tools import parse_arguments
from ctfp_tools import run_ctfp_model as run_model, plot_model
from train_misc import (
    create_regularization_fns,
)
from train_misc import set_cnf_options
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    args = parse_arguments()
    
    #### args ######
    
    args.test_batch_size = 100
    args.num_blocks = 1
    args.save = "experiments/4_Experiment_Sqrt_t/sqrt_t_K2_N3" 
    args.num_workers = 2
    args.layer_type = "concat"
    args.dims = '32,64,64,32'
    args.nonlinearity = "tanh"
    args.data_path = os.path.join(file_path, "data/vol_sqrt_t_mu_sigma_neb.pkl")
    args.activation = "identity"
    args.resume = os.path.join(file_path, args.save , "checkpt_best.pth")
    args.time_change = "M_MGN"
    args.time_change_MGN_K = 2
    args.time_change_MGN_N = 3
    
    args.l1_reg_loss = False
    args.alpha_l1_reg_loss = 1.
    
    args.l2_reg_loss = False
    args.alpha_l2_reg_loss = 1.
    
    ##### args  ######
    
    path_log = os.path.join(file_path, 'eval_time_changed_NF_2.py')
    logger = utils.get_logger(
        logpath=os.path.join(file_path,args.save, "logs_test"), filepath=path_log#os.path.abspath(__file__)
    )

    if args.layer_type == "blend":
        logger.info(
            "!! Setting time_length from None to 1.0 due to use of Blend layers."
        )
        args.time_length = 1.0

    logger.info(args)
    # get deivce
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.use_cpu:
        device = torch.device("cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    test_loader = get_dataset(args, args.test_batch_size)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    aug_model_TC, time_change_func = build_augmented_model_tabular(
        args,
        args.aug_size + args.effective_shape,
        regularization_fns=regularization_fns,
        time_change = args.time_change
    )
    set_cnf_options(args, aug_model_TC)
    logger.info(aug_model_TC)
    
    
    
    # restore parameters
    itr = 0
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        aug_model_TC.load_state_dict(checkpt["state_dict"])
        time_change_func.load_state_dict(checkpt["time_func_state_dict"])
        
    if torch.cuda.is_available() and not args.use_cpu:
        aug_model_TC = torch.nn.DataParallel(aug_model_TC).cuda()
        time_change_func = time_change_func.cuda()
    
    
        
    best_loss = float("inf")
    
    aug_model_TC.eval()
    with torch.no_grad():
        logger.info("Testing...")
        losses = []
        l2_reg_losses = []
        l1_reg_losses = []
        num_observes = []
        for _, x in enumerate(test_loader):
            ## x is a tuple of (values, times, stdv, masks)
            start = time.time()
            # cast data and move to device
            x = map(cvt, x)
            values, times, vars, masks = x
            loss, reg_loss_time_change = run_model(args, aug_model_TC, values, times, vars, masks, time_change_func)
            losses.append(loss.data.cpu().numpy())
            
            l2_reg_losses.append(reg_loss_time_change[0].data.cpu().numpy())
            l1_reg_losses.append(reg_loss_time_change[1].data.cpu().numpy())
            
            num_observes.append(torch.sum(masks).data.cpu().numpy())
        loss = np.sum(np.array(losses) * np.array(num_observes)) / np.sum(num_observes)
        l2_reg_loss = np.mean(l2_reg_losses)
        l1_reg_loss = np.mean(l1_reg_losses)
        
        logger.info("Bit/dim {:.4f} | L2_reg_loss {:.4f} | L1_reg_loss {:.4f}".format(loss, l2_reg_loss, l1_reg_loss))
        