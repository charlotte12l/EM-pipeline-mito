import numpy as np
import torch
import h5py as h
import torch.nn as nn
import torch.nn.functional as F
import os

def writeh5(filename, datasetname, dtarray):
    fid = h.File(filename, 'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()


file_path1 = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Slice4096/for1200/seg_200.h5'
resized1 = np.array(h.File(file_path1, 'r')['main'])[::2,::8,::8]
print(np.shape(resized1))

file_path2 = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Slice4096/for1200/seg_400.h5'
resized2 = np.array(h.File(file_path2, 'r')['main'])[::2,::8,::8]
print(np.shape(resized2))


file_path3 = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Slice4096/for1200/seg_600.h5'
resized3 = np.array(h.File(file_path3, 'r')['main'])[::2,::8,::8]
print(np.shape(resized3))

resized = np.concatenate((resized1,resized2, resized3),axis=0)

print(resized.shape)
#print(np.unique(resized))

#print(np.shape(result))
writeh5('/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Slice4096/for1200/pre1.h5', 'main',resized)