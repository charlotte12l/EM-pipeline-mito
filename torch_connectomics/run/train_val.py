from torch_connectomics.model.loss import *
from torch_connectomics.utils.vis import visualize, visualize_aff
from torch_connectomics.utils.net import *
import numpy as np
import h5py

def train_val(args, train_loader,test_loader, model, model_io_size, pad_size, device, criterion,
          optimizer, scheduler, logger, writer, regularization=None):
    record = AverageMeter()


    # for iteration, (_, volume, label, class_weight, _) in enumerate(train_loader):
    for iteration, batch in enumerate(train_loader):
        model.train()
        if args.task == 22:
            _, volume, seg_mask, class_weight, _, label, out_skeleton = batch
        else:
            _, volume, label, class_weight, _ = batch
        volume, label = volume.to(device), label.to(device)

#        seg_mask = seg_mask.to(device)
        class_weight = class_weight.to(device)
        output = model(volume)

        if regularization is not None:
            loss = criterion(output, label, class_weight) + regularization(output)
        else:
            loss = criterion(output, label, class_weight)
        record.update(loss, args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration, \
                loss.item(), optimizer.param_groups[0]['lr']))

        if iteration % 10000 == 0 and iteration >= 1:
            precision, recall = test(args, test_loader, model, device, model_io_size, pad_size)
            writer.add_scalar('test precision',precision,iteration)
            writer.add_scalar('test recall',recall,iteration)
            
        if iteration % 5 == 0 and iteration >= 1:
            writer.add_scalar('Loss', record.avg, iteration)
            print('[Iteration %d] train_loss=%0.4f lr=%.6f' % (iteration, \
                  record.avg, optimizer.param_groups[0]['lr']))
            scheduler.step(record.avg)
            record.reset()
            if args.task == 0:
                visualize_aff(volume, label, output, iteration, writer)
            elif args.task == 1 or args.task == 2 or args.task == 22:
                visualize(volume, label, output, iteration, writer)
            #print('weight factor: ', weight_factor) # debug
            # debug
            # if iteration < 50:
            #     fl = h5py.File('debug_%d_h5' % (iteration), 'w')
            #     output = label[0].cpu().detach().numpy().astype(np.uint8)
            #     print(output.shape)
            #     fl.create_dataset('main', data=output)
            #     fl.close()

        #Save model
        if iteration % args.iteration_save == 0 or iteration >= args.iteration_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (iteration)))

        # Terminate
        if iteration >= args.iteration_total:
            break    #


def test(args, test_loader, model, device, model_io_size, pad_size, do_eval=True, do_3d=True, model_output_id=None):
    if do_eval:
        # switch to eval mode
        model.eval()
        print('do eval:', do_eval)
    else:
        model.train()
    volume_id = 0
    ww = blend(model_io_size)
    assert(np.min(np.min(np.min(ww)))!=0),"ww=0"
    NUM_OUT = args.out_channel

    result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in test_loader.dataset.input_size]
    weight = [np.zeros(x, dtype=np.float32) for x in test_loader.dataset.input_size]
    print(result[0].shape, weight[0].shape)
    #print(len(result))

    start = time.time()
    with torch.no_grad():
        for i, (pos, volume) in enumerate(test_loader):
            volume_id += args.batch_size
            print('volume_id:', volume_id)
            # print('volume_size:',volume.size())

            # for gpu computing
            volume = volume.to(device)

            if do_3d:
                output = model(volume)
                #output = model(volume)[0] it is for embedding
            else:
                output = model(volume.squeeze(1))

            if args.aux==1:
                output = output[-1]

            if model_output_id is not None:
                output = output[model_output_id]

            sz = tuple([NUM_OUT] + list(model_io_size))
            # print('size:',sz)
            # print('output size:',output.size())
            for idx in range(output.size()[0]):
                st = pos[idx]
                result[st[0]][:, st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], \
                st[3]:st[3] + sz[3]] += output[idx].cpu().detach().numpy().reshape(sz) * np.expand_dims(ww, axis=0)
                weight[st[0]][st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], \
                st[3]:st[3] + sz[3]] += ww

    # print("st:",st)
    end = time.time()
    print("prediction time:", (end - start))
    assert(np.min(np.min(np.min(weight[0])))!=0),"weight=0"

    # print("result len:",len(result))
    # print("result size:",result.shape)
    for vol_id in range(len(result)):
        #assert
        result[vol_id] = result[vol_id] / weight[vol_id]
        dummy_data = result[vol_id]
        # indices_neg = dummy_data < 0
        # dummy_data[indices_neg] = 0
        data = (dummy_data * 255).astype(np.uint8)
        sz = data.shape
        # print('data shape:',data.shape)
        data = data[:,
               pad_size[0]:sz[1] - pad_size[0],
               pad_size[1]:sz[2] - pad_size[1],
               pad_size[2]:sz[3] - pad_size[2]]
        #print('Output shape: ', data.shape)
        seg_eval = (data > 0).astype(int)
        gt = np.array(h5py.File('/n/pfister_lab2/Lab/xingyu/Human/Dataset/test_gt60.h5', 'r')['main'])
        gt = (gt > 0).astype(int)
        # eval
        tp = np.sum(gt & seg_eval)
        fp = np.sum((~gt) & seg_eval)
        tn = np.sum(gt & (~seg_eval))
        print(tp, fp, tn)

        precision = float(tp) / (tp + fp)
        recall = tp / (tp + tn)
        print(precision, recall)

        hf = h5py.File(args.output + 'volume_' + str(vol_id) + '.h5', 'w')
        hf.create_dataset('main', data=data, compression='gzip')
        hf.close()
    return precision, recall
