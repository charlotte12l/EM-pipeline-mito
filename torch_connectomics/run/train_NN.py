import torch

torch.cuda.manual_seed_all(2)
from torch_connectomics.model.loss import *
from torch_connectomics.utils.vis import visualize, visualize_aff
from torch_connectomics.utils.net import *
import numpy as np

class NN(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x_a, x_b):
        
        x_a = (x_a > 0.5).float()  # get a inital saliency map
        # print(torch.unique(x_a))

        b, c, d, h, w = x_b.shape

        x_b = x_b.contiguous().view(b, c, d * h * w)  # feature map

        x_a = x_a.contiguous().view(b, 1, d * h * w)

        fore_sample = (x_a * x_b).cuda()
        # print(fore_sample.shape)
        fore_ac = torch.sum(fore_sample, 2) / torch.sum(x_a, 2)  # b,c
        bkg_sample = ((1 - x_a) * x_b).cuda()
        # print(bkg_sample.shape)
        bkg_ac = torch.sum(bkg_sample, 2) / torch.sum(1 - x_a, 2)  # b,c

        fore_ac = fore_ac.unsqueeze(2)
        bkg_ac = bkg_ac.unsqueeze(2)

        fore_d = torch.norm(x_b-fore_ac,p=2,dim=1).cuda()
        bkg_d = torch.norm(x_b-bkg_ac,p=2,dim=1).cuda()

        fore_d = fore_d.contiguous().view(b, 1, d,  h, w)
        bkg_d = bkg_d.contiguous().view(b, 1, d,  h, w)

      # mask!
        fore_p = torch.exp(-fore_d)/(torch.exp(-fore_d) + torch.exp(-bkg_d))
        assert torch.exp(-fore_d)!=0
        assert torch.exp(-bkg_d)!=0
        fore_p = fore_p.cuda()
        return fore_p 


def train_NN(args, train_loader, model, device, criterion,
          optimizer, scheduler, logger, writer, regularization=None):
    record = AverageMeter()
    model.train()
    print('train_NN')
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
        
        #print('begin NN:')
        NNC = NN()
        output = NNC(seg_out, ins_out)
        
        output = output.cuda()
        loss = criterion(output,label,class_weight)

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


