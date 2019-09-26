import torch

torch.cuda.manual_seed_all(2)
from torch_connectomics.model.loss import *
from torch_connectomics.utils.vis import visualize, visualize_aff
from torch_connectomics.utils.net import *
import numpy as np


def train_2d(args, train_loader, model, device, criterion,
          optimizer, scheduler, logger, writer, regularization=None):
    record = AverageMeter()
    model.train()

    # for iteration, (_, volume, label, class_weight, _) in enumerate(train_loader):
    for iteration, batch in enumerate(train_loader):
        iteration = iteration + args.iteration_begin
        # print('begin:',iteration)
        if args.task == 22:
            # _, volume, seg_mask, class_weight, _, label, out_skeleton, out_valid = batch
            if args.valid_mask is None:
                _, volume, seg_mask, class_weight, _, label, out_skeleton = batch
            else:
                _, volume, seg_mask, class_weight, _, label, out_skeleton, out_valid = batch
                out_valid = out_valid.to(device)
        else:
            _, volume, label, class_weight, _ = batch
        volume, label = volume.to(device), label.to(device)
        #print(volume.shape,label.shape)
        volume = volume.squeeze(2)
        label = label.squeeze(2)
        #        seg_mask = seg_mask.to(device)
        class_weight = class_weight.to(device)
        class_weight = class_weight.squeeze(2)
        if args.ebd == 1:
            p = 5
            noisy_label = (label.cpu().numpy() + np.random.binomial(1, float(p) / 100.0, (320, 320))) % 2
            noisy_label = torch.Tensor(noisy_label).cuda()
            # print('getting:')
            output = model(volume, noisy_label)
            # print('got:')
        else:
            output = model(volume)

        if args.task == 22 and args.valid_mask is not None:
            class_weight = class_weight * out_valid

        if regularization is not None:
            loss = criterion(output, label, class_weight) + regularization(output)
        else:
            loss = criterion(output, label, class_weight)
            # print('loss:')
        record.update(loss, args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration, \
                                                                 loss.item(), optimizer.param_groups[0]['lr']))

        if iteration % 10 == 0 and iteration >= 1:
            writer.add_scalar('Loss', record.avg, iteration)
            print('[Iteration %d] train_loss=%0.4f lr=%.6f' % (iteration, \
                                                               record.avg, optimizer.param_groups[0]['lr']))
            scheduler.step(record.avg)
            record.reset()
            if args.task == 0:
                visualize_aff(volume, label, output, iteration, writer)
            elif args.task == 1 or args.task == 2 or args.task == 22:
                visualize(volume, label, output, iteration, writer, aux=args.aux)
            # print('weight factor: ', weight_factor) # debug
            # debug
            # if iteration < 50:
            #     fl = h5py.File('debug_%d_h5' % (iteration), 'w')
            #     output = label[0].cpu().detach().numpy().astype(np.uint8)
            #     print(output.shape)
            #     fl.create_dataset('main', data=output)
            #     fl.close()

        # Save model
        if iteration % args.iteration_save == 0 or iteration >= args.iteration_total:
            torch.save(model.state_dict(), args.output + ('/volume_%d.pth' % (iteration)))

        # Terminate
        if iteration >= args.iteration_total:
            break  #


