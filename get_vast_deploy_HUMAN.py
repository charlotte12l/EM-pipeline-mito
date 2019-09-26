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

folder = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/for1200/0/0/'

def writeh5(filename, datasetname, dtarray):
    fid = h.File(filename, 'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()

def seg2Vast(seg):
    return np.stack([seg//65536, seg//256, seg%256],axis=2).astype(np.uint8)

def get_whole_slice(z):
    whole = np.zeros((12320, 12320))
    sliceN = z%192
    maskN = math.floor(z/192)*192
    for x in range(0,12320,1540):
        for y in range(0,12320,1540):
            path = folder + str(x) +'/' +str(y)+'/' + str(maskN) +'/mask.h5'
            whole[y:y+1540,x:x+1540]=np.array(h.File(path, 'r')['main'])[sliceN]
    #new_whole1 = whole[np.newaxis, :, :]
    whole = whole[np.newaxis, :, :].astype(np.uint8)
    heatmap_path = folder + 'WholeSlice/Heatmaps/whole_heatmap_' + str(z)+'.h5'
    writeh5(heatmap_path, 'main', whole)
    print('Slice Done:',z)


def get_ws(z):
    #for z in range(200,1240,200):
    energy_200 = np.zeros((200,4096,4096))
    for k in range(0,200,20):
        id_20 = k+z
        heatmap_20_path = folder + str(id_20) +'/mask.h5'
        energy_20 = np.array(h.File(heatmap_20_path, 'r')['main'])
        energy_200[k:k+20,:,:] = energy_20

    del energy_20

    ##CC
    seg = get_seg(energy_200, None, 16)

    nlabels, count = np.unique(seg, return_counts=True)#count return the times

    indices = np.argsort(count)
    nlabels = nlabels[indices]
    count = count[indices]

    least_index = np.where(count >= 1100)[0][0]

    count = count[least_index:]
    nlabels = nlabels[least_index:]

    rl = np.arange(seg.max() + 1).astype(seg.dtype)

    for i in range(seg.max() + 1):
        if i not in nlabels:
            rl[i] = 0

    seg = rl[seg]

    del energy_200

	## Watershed
        #reload the heatmap 200
    energy_200 = np.zeros((200,4096,4096)).astype(np.float32)
    for k in range(0,200,20):
        id_20 = k+z
        heatmap_20_path = folder + str(id_20) +'/mask.h5'
        energy_20 = np.array(h.File(heatmap_20_path, 'r')['main']).astype(np.float32)
        energy_200[k:k+20,:,:] = energy_20

    del energy_20
            
        
    threshold = 150

    energy_thres = energy_200 - threshold
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

    labels = watershed(-energy_200, mask=mask, markers=markers)

    segws_path = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Slice4096/for1200/seg_1200.h5'
    writeh5(segws_path, 'main', labels)

     for j in range(200):
             ## save to png files for VAST proofreading
         label = labels[j]
         id = "%04d"%(j+z)
         png = seg2Vast(label)
         filedir = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Slice4096/for1200/ToVAST/'
         if not os.path.exists(filedir):
             os.makedirs(filedir)
         file_path = filedir + str(id)+'_'+'tr1-tc1.png'
         imwrite(file_path, png)

         print('Vast Done:',id)

def main_vast(start, jobId, jobNum):
    #get_whole_slice(start,jobId,jobNum)
    for z in range(start+jobId,200,jobNum):
        #get_whole_slice(z)
        get_ws(z)


if __name__ == "__main__":
    #start = int(sys.argv[1])
    get_ws(1000)
