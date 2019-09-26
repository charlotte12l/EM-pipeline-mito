import torch

torch.cuda.manual_seed_all(2)
from torch_connectomics.model.loss import *
from torch_connectomics.utils.vis import visualize, visualize_aff
from torch_connectomics.utils.net import *
import numpy as np



def train_dsc(args, train_loader, model, device, criterion,
          optimizer, scheduler, logger, writer, regularization=None):
    record = AverageMeter()
    model.train()
    print('train_dsc')
    # for iteration, (_, volume, label, class_weight, _) in enumerate(train_loader):
    for iteration, batch in enumerate(train_loader):
        iteration = iteration + args.iteration_begin
        #print('begin:',iteration)
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
        #        seg_mask = seg_mask.to(device)
        class_weight = class_weight.to(device)

        seg_out,ins_out = model(volume)

        ins_criterion = DiscriminativeLoss(device)

        label_0 = 1 - label
        ins_label = torch.cat((label_0, label), 1)
        bs, ch, d, h, w = ins_out.shape
        ins_label = ins_label.contiguous().view(bs, 2, d * h * w)
        ins_out = ins_out.contiguous().view(bs, ch, d * h * w)
        
        ins_loss = ins_criterion(ins_out, ins_label,[2]*bs)

        seg_loss = criterion(seg_out,label,class_weight)

        loss = ins_loss + seg_loss
        record.update(loss, args.batch_size)

        output = seg_out
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


