# Inference on Large-Scale Datasets
from __future__ import division, print_function
import os, sys

sys.path.append('/n/pfister_lab2/Lab/xingyu/Human/pytorch_connectomics')

import torch
import torch.nn as nn
import numpy as np
import h5py, yaml, json
from matplotlib.pyplot import imread
import itertools

# model
from torch_connectomics.utils.net import *
from torch_connectomics.model.model_zoo.unetv3 import unetv3
from torch_connectomics.data.dataset import AffinityDataset, SynapseDataset, SynapsePolarityDataset, MitoDataset, \
    MitoSkeletonDataset
from torch_connectomics.data.utils import collate_fn, collate_fn_test, collate_fn_skel

# import argparse
# parser = argparse.ArgumentParser(description='Specify model inference arguments.')

# args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = setup_model(args, device, exact=True)

# get parameter
D0 = '/n/pfister_lab2/Lab/xingyu/Human/pytorch_connectomics/torch_connectomics/run/configs/'
param = yaml.load(open(D0 + 'run_human-1-1-1200.yaml'))
bfly_path = D0 + "bfly_1240_zpad0_mip1.json"
bfly_db = json.load(open(bfly_path))

p_vol = param['data']
tile_sz = p_vol['tile-size']
z0 = p_vol['z0'];
z1 = p_vol['z1']
y0 = p_vol['y0'];
y1 = p_vol['y1']
x0 = p_vol['x0'];
x1 = p_vol['x1']

# input volume: z,y,x
p_aff = param['aff']
in_sz = [p_aff['mz'], p_aff['my'], p_aff['mx']]  # model output
vol_sz = [p_aff['vz'], p_aff['vy'], p_aff['vx']]  # read more chunk
pad_sz = [p_aff['pz'], p_aff['py'], p_aff['px']]  # read more chunk

# classifier
#Dp = '/n/pfister_lab2/Lab/xingyu/Human/configs/'
classifier_path = D0 + 'volume_49000.pth'
print(classifier_path)
Do = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/'
destination = Do + "for1200/"
print(destination)


# generate volume for prediction
def get_dataset(dataset_dict, x0p, x1p, y0p, y1p, z0p, z1p, tile_sz):
    # no padding at the boundary
    print('original: ', z0p, z1p, y0p, y1p, x0p, x1p)
    boundary = [z0p < z0, z1p > z1, y0p < y0, y1p > y1, x0p < x0, x1p > x1]
    print('touch boundary? ', boundary)
    pad_size = [pad_sz[0], pad_sz[0],
                pad_sz[1], pad_sz[1],
                pad_sz[2], pad_sz[2]]
    pad_need = list(np.array(boundary).astype(int) * np.array(pad_size))
    # print(pad_need)
    z0p = z0p + pad_need[0]
    z1p = z1p - pad_need[1]
    y0p = y0p + pad_need[2]
    y1p = y1p - pad_need[3]
    x0p = x0p + pad_need[4]
    x1p = x1p - pad_need[5]

    print('adjusted: ', z0p, z1p, y0p, y1p, x0p, x1p)
    result = np.zeros((z1p - z0p, y1p - y0p, x1p - x0p), np.uint8)
    c0 = x0p // tile_sz  # floor
    c1 = (x1p + tile_sz - 1) // tile_sz  # ceil
    r0 = y0p // tile_sz
    r1 = (y1p + tile_sz - 1) // tile_sz
    for z in range(z0p, z1p):
        for row in range(r0, r1):
            for column in range(c0, c1):
                pattern = dataset_dict["sections"][z]
                path = pattern.format(row=row + 1, column=column + 1)
                patch = imread(path, 0)
                xp0 = column * tile_sz
                xp1 = (column + 1) * tile_sz
                yp0 = row * tile_sz
                yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x0a = max(x0p, xp0)
                    x1a = min(x1p, xp1)
                    y0a = max(y0p, yp0)
                    y1a = min(y1p, yp1)
                    result[z - z0p, y0a - y0p:y1a - y0p, x0a - x0p:x1a - x0p] = patch[y0a - yp0:y1a - yp0,
                                                                                x0a - xp0:x1a - xp0]

    if True in boundary:
        result = np.pad(result,
                        ((pad_need[0], pad_need[1]),
                         (pad_need[2], pad_need[3]),
                         (pad_need[4], pad_need[5])), 'reflect')
    print('input shape: ', result.shape)
    return result


def writeh5(filename, datasetname, dtarray):
    fid = h5py.File(filename, 'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()


def get_dest_path(x, y, z):
    return os.path.join(destination, str(x), str(y), str(z))


def process_message(classifier, x0a, x1a, y0a, y1a, z0a, z1a):
    p1 = get_dest_path(x0a, y0a, z0a)
    redo = False
    path = os.path.join(p1, 'mask.h5')
    if not os.path.exists(path):
        redo = True

    if redo:

        # rh_logger.logger.report_event("process: %d,%d,%d" % (x0,y0,z0))
        aug_index = 0
        data = get_dataset(bfly_db, x0a - pad_sz[2], x1a + pad_sz[2],
                           y0a - pad_sz[1], y1a + pad_sz[1],
                           z0a - pad_sz[0], z1a + pad_sz[0], tile_sz)
        # rint(np.max(data),np.min(data))
        data = data / 255.0  # normalize image to (0,1)

        print('data shape: ', data.shape)
        # out = [None] * 16 #different type of simple augmentation
        # for xflip, yflip, zflip, transpose in itertools.product(
        #                 (False, True), (False, True), (False, True), (False, True)):
        out = [None]
        for xflip, yflip, zflip, transpose in itertools.product(
                (False,), (False,), (False,), (False,)):

            print((aug_index, xflip, yflip, zflip, transpose))
            volume = data.copy()
            if xflip:
                volume = volume[:, :, ::-1]
            if yflip:
                volume = volume[:, ::-1, :]
            if zflip:
                volume = volume[::-1, :, :]
            if transpose:
                volume = volume.transpose(0, 2, 1)
            # synapse: 3*z*y*x

            model_io_size = np.array(in_sz, dtype=int)
            # print('volume shape: ', volume.shape)
            # print('model_io_size: ', model_io_size)
            # dataset = SynapseDataset(volume=[volume], label=None, vol_input_size=model_io_size,
            # vol_label_size=None, sample_stride=model_io_size/2,
            # data_aug=None, mode='test')
            dataset = MitoSkeletonDataset(volume=[volume], label=None, sample_input_size=model_io_size,
                                          sample_label_size=None, sample_stride=model_io_size / 2,
                                          augmentor=None, mode='test')

            test_loader = torch.utils.data.DataLoader(
                dataset, batch_size=28, shuffle=False, collate_fn=collate_fn_test,
                num_workers=0, pin_memory=True)

            ww = blend(sz=model_io_size)
            sz = tuple([1] + list(model_io_size))
            # print(tuple([3]+list(volume.shape)))
            result = np.zeros(tuple([1] + list(volume.shape))).astype(np.float32)  # 1, 200,1700,1700
            weight = np.zeros(volume.shape).astype(np.float32)
            print('weight shape:', weight.shape)

            with torch.no_grad():
                for _, (pos, vol_input) in enumerate(test_loader):
                    vol_input = vol_input.to(device).float()  # b, c, z, y, x
                    # output = torch.zeros(vol_input.shape, dtype=torch.float32)
                    # # print('output shape:',output.shape)
                    # for i in range(vol_input.size()[2]):
                    #     one_slice = vol_input[:, :, i, :, :]  # b, c , y,x
                    #     one_slice_output, so1, so2, so3 = classifier(one_slice)
                    #     # print('one out shape:',one_slice_output.shape)
                    #     output[:, :, i, :, :] = one_slice_output  # b, c, y, x

                    output = classifier(vol_input) # (b, 3, z, y, x)

                    for idx in range(output.size()[0]):
                        st = pos[idx]
                        result[:, st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], \
                        st[3]:st[3] + sz[3]] += output[idx].cpu().detach().numpy() * np.expand_dims(ww, axis=0)
                        weight[st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], \
                        st[3]:st[3] + sz[3]] += ww

                    # rint(j,':finished')

            result = result / np.expand_dims(weight, axis=0)
            result[result < 0] = 0
            result = (result * 255).astype(np.uint8)

            if transpose:  # swap x-/y-affinity
                result = result.transpose(0, 1, 3, 2)
            if zflip:
                result = result[:, ::-1, :, :]
            if yflip:
                result = result[:, :, ::-1, :]
            if xflip:
                result = result[:, :, :, ::-1]
            out[aug_index] = result
            aug_index += 1

        if len(out) == 1:
            final = out[0]
        else:
            final = np.mean(np.stack(out, axis=0), axis=0).astype(np.uint8)
        # inal[final < 128] = 0
        final = final[0, pad_sz[0]:-pad_sz[0], pad_sz[1]:-pad_sz[1], pad_sz[2]:-pad_sz[2]]
        print('output size:', final.shape)
        path = os.path.join(p1, 'mask.h5')
        writeh5(path, 'main', final)


def main_db(jobId, jobNum):
    # rh_logger.logger.start_process("Worker %d" % jobId, "starting", [])

    # classifier = unet_SE_synBN(in_num=1, out_num=3, filters=[32,64,128,256], aniso_num=2)
    # classifier = fpn1(in_num=1, out_num=3, filters=[32,64,128,256])
    classifier = unetv3(in_channel=1, out_channel=1, act='tanh')
    print('classifier: ', classifier.__class__.__name__)
    classifier = DataParallelWithCallback(classifier, range(2))
    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(classifier_path))

    count = 0
    for x0a in range(x0, x1, vol_sz[2]):
        x1a = x0a + vol_sz[2]
        for y0a in range(y0, y1, vol_sz[1]):
            y1a = y0a + vol_sz[1]
            for z0a in range(z0, z1, vol_sz[0]):
                z1a = z0a + vol_sz[0]
                dir_path = get_dest_path(x0a, y0a, z0a)
                if count % jobNum == jobId:
                    if not os.path.exists(os.path.join(dir_path, 'mask.h5')):  # no prediction yet
                        print('start prediction')
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        process_message(classifier, x0a, x1a, y0a, y1a, z0a, z1a)
                    print('prediction for volume %d is finished!\n' % (count))
                count += 1

            # def check_done():


#     count = 0
#     for x0a in range(x0, x1, vol_sz[2]):
#         for y0a in range(y0, y1, vol_sz[1]):
#             for z0a in range(z0, z1, vol_sz[0]):
#                 p1 = get_dest_path(x0a, y0a, z0a)
#                 count += 1
#                 path = os.path.join(p1, 'mask.h5')
#                 if not os.path.exists(path):
#                     print("undone: %d, %d, %d" % (x0a, y0a, z0a))
#     print('total jobs: ',count)

if __name__ == "__main__":
    # python classify4-jwr_20um.py 0 0,1,2,3,4,5,6,7,9
    jobId = int(sys.argv[1])
    jobNum = int(sys.argv[2])
    main_db(jobId, jobNum)  # single thread
    # check_done() # single thread
