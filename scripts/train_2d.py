import os, sys

sys.path.append('/n/pfister_lab2/Lab/xingyu/pytorch_connectomics')
import numpy as np
import h5py, time, itertools, datetime

import torch

torch.cuda.manual_seed_all(2)
from torch_connectomics.utils.net import *
from torch_connectomics.run.train_2d import train_2d
from torch_connectomics.model.loss import WeightedMSE, WeightedBCE, BinaryReg, AuxBCELoss, AuxMSELoss, AuxOhemMSELoss, \
    AuxOhemBCELoss, AuxOhemMSELoss2
from torch_connectomics.model.model_zoo import unet2d_ebd

def setup_model_2d(args, device, exact=True, size_match=True):
    model = unet2d_ebd()
    print('model: ', model.__class__.__name__)
    if args.num_gpu > 1:
        model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    model = model.to(device)

    if bool(args.load_model):
        print('Load pretrained model:')
        print(args.pre_model)
        if exact:
            model.load_state_dict(torch.load(args.pre_model))
        else:
            pretrained_dict = torch.load(args.pre_model)
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            if size_match:
                model_dict.update(pretrained_dict)
            else:
                for param_tensor in pretrained_dict:
                    if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                        model_dict[param_tensor] = pretrained_dict[param_tensor]
                        # 3. load the new state dict
            model.load_state_dict(model_dict)

    return model


def main():
    args = get_args(mode='train')

    print('0. initial setup')
    model_io_size, device = init(args)
    logger, writer = get_logger(args)

    print('1. setup data')
    train_loader = get_input(args, model_io_size, 'train')

    print('2.0 setup model')
    model = setup_model_2d(args, device)
    print('use aux:', args.aux)

    print('2.1 setup loss function')
    if args.aux is True:
        if args.task == 22:
            criterion = AuxMSELoss()  # OHEM on
            print('use AuxMSELoss on weighted aux')
        else:
            criterion = AuxOhemBCELoss(0.5, 10000)
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
    print('use ebd:', args.ebd)
    train_2d(args, train_loader, model, device, criterion, optimizer, scheduler, logger, writer)

    print('5. finish training')
    logger.close()
    writer.close()


if __name__ == "__main__":
    main()

