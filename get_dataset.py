from imageio import imread
import numpy as np
import h5py as h
import yaml, json

def vast2Seg(seg):
    # convert to 24 bits
    return seg[:,:,0].astype(np.uint32)*65536+seg[:,:,1].astype(np.uint32)*256+seg[:,:,2].astype(np.uint32)

def writeh5(filename, datasetname, dtarray):
    fid = h.File(filename, 'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()

def readh5(filename, datasetname='main'):
    return np.array(h.File(filename, 'r')[datasetname])


# a small test
#base = './Dataset/proofreading/_s'
#img = imread('./Dataset/proofreading/_s0000.png')
#print(np.shape(img)) #4096,4096,3
#print(np.unique(img)) #dtyepe=uint8


# convert .png proofreading to .h5 groundtruth
base = './Dataset/proofreading/_s'
all_gt = np.zeros((200,4096,4096))
for z in range(200):
    id = "%04d"%z
    path = base+str(id)+'.png'
    img = imread(path)
    all_gt[z,:,:] = vast2Seg(img)
all_gt_path = '/n/pfister_lab2/Lab/xingyu/Human/Dataset/all_gt200.h5'
writeh5(all_gt_path, 'main', all_gt)
# please check the groudtruth h5 file is uint32, or it would be a problem when training

# convert .png data to .h5 training volm
D0 = '/n/pfister_lab2/Lab/xingyu/Human/pytorch_connectomics/torch_connectomics/run/configs/'
param = yaml.load(open(D0 + 'run_human-1-1-200.yaml'))
bfly_path = D0 + "bfly_1240_zpad0_mip1.json"
bfly_db = json.load(open(bfly_path))

from matplotlib.pyplot import imread # this is really important, imread here is different from imread before
train_volm = np.zeros((200,4096,4096))
for z in range(200):
    pattern = bfly_db["sections"][z]
    path = pattern.format(row=1, column = 1)
    patch = imread(path, 0)
    train_volm[z,:,:] = patch
train_volm_path = '/n/pfister_lab2/Lab/xingyu/Human/Dataset/train_volm200.h5'
writeh5(train_volm_path, 'main', train_volm)
# please check the train_volm h5 file is uint8, or it would be a problem when training
