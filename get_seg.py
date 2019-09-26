import numpy as np
import h5py as h
import math
import os
import sys
#sys.path.append('/home/xingyu/anaconda3/envs/pytorch/lib/python3.7/site-packages')


from numpy import ma
import h5py as h

from scipy import ndimage as ndi
from skimage.morphology import watershed
from scipy.ndimage import label as label_scipy
#from scipy.misc.pilutil import imsave
from imageio import imwrite
from helper import *
import time

folder = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Training/a18'
heatmap_20_path = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Training/a18volume_0.h5'
def writeh5(filename, datasetname, dtarray):
    fid = h.File(filename, 'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()

def seg2Vast(seg):
    return np.stack([seg//65536, seg//256, seg%256],axis=2).astype(np.uint8)


def get_ws():

    energy = np.array(h.File(heatmap_20_path, 'r')['main'])[0]

    #energy = energy[np.newaxis, :, :]


    ##CC
    seg = get_seg(energy, None, 16)

    nlabels, count = np.unique(seg, return_counts=True)#count return the times

    indices = np.argsort(count)
    nlabels = nlabels[indices]
    count = count[indices]

    least_index = np.where(count >= 1000)[0][0]

    count = count[least_index:]
    nlabels = nlabels[least_index:]

    rl = np.arange(seg.max() + 1).astype(seg.dtype)

    for i in range(seg.max() + 1):
        if i not in nlabels:
            rl[i] = 0

    seg = rl[seg]

    # segcc_path = folder + 'WholeSlice/SegCC/whole_segcc_' + str(z) + '.h5'
    # writeh5(segcc_path,'main',seg)

    ## Watershed
    energy = np.array(h.File(heatmap_20_path, 'r')['main'])[0].astype(np.float32)

    threshold = 150

    energy_thres = energy - threshold
    markers_unlabelled = (energy_thres > 0).astype(int)
    markers, ncomponents = label_scipy(markers_unlabelled)
    labels_d, count_d = np.unique(markers, return_counts=True)

    rl = np.arange(markers.max() + 1).astype(markers.dtype)
    pixel_threshold = 100

    for i in range(len(labels_d)):
        if count_d[i] < pixel_threshold:
            rl[labels_d[i]] = 0

    markers = rl[markers]

    mask = (seg > 0).astype(int)# uses cc

    labels = watershed(-energy, mask=mask, markers=markers)

    segws_path = folder+ 'seg_' + '_'+str(threshold)+ '.h5'
    writeh5(segws_path, 'main', labels)

    seg_eval = (labels>0).astype(int)
    gt = np.array(h.File('/n/pfister_lab2/Lab/xingyu/Human/Dataset/test_gt60.h5', 'r')['main'])
    gt_eval = (gt>0).astype(int)
    
    print(np.shape(seg_eval))
    print(np.shape(gt_eval))
    tp = np.sum(gt_eval&seg_eval)
    fp = np.sum((~gt_eval)&seg_eval)
    tn = np.sum(gt_eval&(~seg_eval))
    print(tp,fp,tn)
    
    precision = float(tp)/(tp+fp)
    recall = tp/(tp+tn)
    print(precision,recall)

def main_vast(start, jobId, jobNum):
    #get_whole_slice(start,jobId,jobNum)
    for z in range(start+jobId,200,jobNum):
        #get_whole_slice(z)
        get_ws(z)


if __name__ == "__main__":
    get_ws()