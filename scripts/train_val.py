import os,sys
sys.path.append('/n/pfister_lab2/Lab/xingyu/pytorch_connectomics')
import numpy as np
import h5py, time, itertools, datetime

import torch
torch.cuda.manual_seed_all(2)
from torch_connectomics.utils.net import *
from torch_connectomics.run.train_val import train_val
from torch_connectomics.model.loss import WeightedMSE, WeightedBCE, BinaryReg, AuxBCELoss, AuxMSELoss,AuxOhemMSELoss,AuxOhemBCELoss


def main():
    args = get_args()
    print('0. initial setup')
    model_io_size, device = init(args)
    logger, writer = get_logger(args)

    print('1. setup data')
    train_loader = get_input(args, model_io_size, 'train')
    test_loader = get_input(args, model_io_size, 'test')

    print('2.0 setup model')
    model = setup_model(args, device)
    print('use aux:',args.aux)
    
    print('2.1 setup loss function')
    if args.aux is True:
        if args.task == 22:
            criterion = AuxMSELoss() # OHEM on 
            print('use AuxMSELoss on weighted aux')
        else:
            criterion = AuxOhemBCELoss(0.5,10000)
    else:
        if args.task == 22:
            criterion = WeightedMSE()
            print('use no aux')
        else:
            criterion = WeightedBCE()  
    regularization = BinaryReg(alpha=10.0)
    

    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                           patience=1000, verbose=False, threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0,
                                                           min_lr=1e-7, eps=1e-08)

    print('4. start training')
    pad_size = model_io_size // 2  # 50% overlap
    print('Pad size:{}'.format(pad_size))
    train_val(args, train_loader,test_loader, model, model_io_size, pad_size, device, criterion, optimizer, scheduler, logger, writer)

    print('5. finish training')
    logger.close()
    writer.close()

def test(args):
    print('0. initial setup')
    model_io_size, device = init(args)
    print('model I/O size:', model_io_size)

    print('1. setup data')


    print('2. setup model')
    model = setup_model(args, device, exact=True)
    # import pdb;pdb.set_trace()

    print('3. start testing')
    pad_size = model_io_size // 2  # 50% overlap
    print('Pad size:{}'.format(pad_size))
    test(args, test_loader, model, device, model_io_size, pad_size, do_eval=True)

    print('4. finish testing')


if __name__ == "__main__":
    main()

