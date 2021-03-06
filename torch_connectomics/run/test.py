import numpy as np
from torch_connectomics.utils.net import *

def test(args, test_loader, model, device, model_io_size, pad_size, do_eval=True, do_3d=True, model_output_id=None):
    if do_eval:
        # switch to eval mode
        model.eval()
        print('do eval:',do_eval)
    else:
        model.train()
    volume_id = 0
    ww = blend(model_io_size)
    NUM_OUT = args.out_channel

    result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in test_loader.dataset.input_size]
    weight = [np.zeros(x, dtype=np.float32) for x in test_loader.dataset.input_size]
    print(result[0].shape, weight[0].shape)
    print(len(result))

    start = time.time()
    with torch.no_grad():
        for i, (pos, volume) in enumerate(test_loader):
            volume_id += args.batch_size
            print('volume_id:', volume_id)
            #print('volume_size:',volume.size())

            # for gpu computing
            volume = volume.to(device)
      
            if do_3d:
                output = model(volume)
                #output = model(volume)[0]
            else:
                output = model(volume.squeeze(1))

            if args.aux ==1 :
                output = output[-1]
                
            if model_output_id is not None:
                output = output[model_output_id]

            sz = tuple([NUM_OUT]+list(model_io_size))
            #print('size:',sz)
            #print('output size:',output.size())
            for idx in range(output.size()[0]):
                st = pos[idx]
                result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                st[3]:st[3]+sz[3]] += output[idx].cpu().detach().numpy().reshape(sz) * np.expand_dims(ww, axis=0)
                weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                st[3]:st[3]+sz[3]] += ww

    #print("st:",st)
    end = time.time()
    print("prediction time:", (end-start))

    #print("result len:",len(result))
    #print("result size:",result.shape)
    for vol_id in range(len(result)):
        result[vol_id] = result[vol_id] / weight[vol_id]
        dummy_data = result[vol_id]
        indices_neg = dummy_data < 0 ## !!!! this really matters
        dummy_data[indices_neg] = 0
        data = (dummy_data*255).astype(np.uint8)
        sz = data.shape
        #print('data shape:',data.shape)
        data = data[:,
                    pad_size[0]:sz[1]-pad_size[0],
                    pad_size[1]:sz[2]-pad_size[1],
                    pad_size[2]:sz[3]-pad_size[2]]
        print('Output shape: ', data.shape)
        hf = h5py.File(args.output+'volume_'+str(vol_id)+'.h5','w')
        hf.create_dataset('main', data=data, compression='gzip')
        hf.close()