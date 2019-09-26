import os,sys
sys.path.append('/n/pfister_lab2/Lab/xingyu/pytorch_connectomics')
import numpy as np
import h5py, time, itertools, datetime

import torch
torch.cuda.manual_seed_all(2)
from torch_connectomics.utils.net import *
from torch_connectomics.run.train_dsc import train_dsc
from torch_connectomics.model.loss import WeightedMSE, WeightedBCE, BinaryReg, AuxBCELoss, AuxMSELoss,AuxOhemMSELoss,AuxOhemBCELoss,AuxOhemMSELoss2

def main():
    args = get_args(mode='train')
    
    print('0. initial setup')
    model_io_size, device = init(args) 
    logger, writer = get_logger(args)
    
    print('1. setup data')
    train_loader = get_input(args, model_io_size, 'train')

    print('2.0 setup model')
    model = setup_model(args, device, exact=False)
    print('use aux:',args.aux)
    
    print('2.1 setup loss function')

    criterion = WeightedBCE()  
    regularization = BinaryReg(alpha=10.0)
 
    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                patience=1000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, 
                min_lr=1e-7, eps=1e-08)
    

    print('4. start training')

    train_dsc(args, train_loader, model, device, criterion, optimizer, scheduler, logger, writer)
  
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()

