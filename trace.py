import h5py as h
import numpy as np
from imageio import imwrite
import os

def readh5(filename, datasetname='main'):
    return np.array(h.File(filename, 'r')[datasetname])


def writeh5(filename, datasetname, dtarray):
    fid = h.File(filename, 'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()


def seg2Vast(seg):
    return np.stack([seg // 65536, seg // 256, seg % 256], axis=2).astype(np.uint8)


def get_bb_label2d_v2(seg, uid=None):
    sz = seg.shape
    assert len(sz) == 2
    if uid is None:  # get uid list
        uid = np.unique(seg)
        uid = uid[uid > 0]

    out = np.zeros((1 + uid.max(), 5), dtype=np.uint32)
    out[:, 0] = np.arange(out.shape[0])
    out[:, 1] = sz[0]
    out[:, 3] = sz[1]
    # for each row
    rids = np.where((seg > 0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[rid])  # find ids in each row
        sid = sid[sid > 0]
        out[sid, 1] = np.minimum(out[sid, 1], rid)  # xmin
        out[sid, 2] = np.maximum(out[sid, 2], rid)  # xmax

    cids = np.where((seg > 0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, cid])
        sid = sid[sid > 0]
        out[sid, 3] = np.minimum(out[sid, 3], cid)  # ymin
        out[sid, 4] = np.maximum(out[sid, 4], cid)  # ymax

    return out[uid]


def getPfSegChunkIou(seg0, seg1):
    ui, uc = np.unique(seg0, return_counts=True)
    ui1, uc1 = np.unique(seg1, return_counts=True)

    uc = uc[ui > 0]  # omit background
    ui = ui[ui > 0]  # omit background

    uc1[ui1 == 0] = 0  # label background count = 0
    out = np.zeros((1 + len(ui), 5), np.uint32)
    out[1:, 0] = ui
    out[1:, 2] = uc
    # for i in range(1,2):
    for i in range(1, out.shape[0]):
        if i % 200 == 0:
            print(i, out.shape[0])
        # use bbox to speed things up
        s0 = ui[i - 1]
        # bb = get_bb(seg0==s0)
        bb = bb0_2d[bb0_2d[:, 0] == s0][0]
        mid, mc = np.unique(seg1[bb[1]:bb[2] + 1, bb[3]:bb[4] + 1][ \
                                seg0[bb[1]:bb[2] + 1, bb[3]:bb[4] + 1] == s0], return_counts=True)
        mc[mid == 0] = 0
        aid = mid[np.argmax(mc)]
        if not hasattr(aid, "__len__") or len(aid) > 0:
            if hasattr(aid, "__len__") and len(aid) > 1:
                aid = aid[0]
            out[i, 1] = aid
            out[i, 3] = uc1[ui1 == aid]
            out[i, 4] = mc.max()
    return out




#for i in range(200, 1000, 200):
    # get bounding box for the previous chunk
i = 0

# the previous seg
seg0_path = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Slice4096/for1200/seg_200.h5'
seg0 = readh5(seg0_path)[199, :, :]
print(np.shape(seg0))
# the next seg
seg1_path = '/n/pfister_lab2/Lab/xingyu/Human/Dataset/seg_400.h5'
seg1 = readh5(seg1_path)[0, :, :].astype(np.uint32)
print(np.shape(seg1))

bb0_2d = get_bb_label2d_v2(seg0)

iou = getPfSegChunkIou(seg0, seg1)[1:]
print(np.shape(iou))


chunk0 = readh5(seg0_path)
mid = chunk0.max()  # max id
print(mid)
#del chunk0, seg0, seg1

chunk1 = readh5(seg1_path).astype(np.uint32)
print(chunk1.dtype)
print(np.shape(chunk1))

# iou[:,0]: id from prev chunk
# iou[:,1]: id from next chunk
# iou[:,2]: seg size for prev chunk seg id
# iou[:,3]: seg size for next chunk seg id
# iou[:,4]: seg size for intersection


iou_gid = (iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3])) >= 0.2
uid1 = np.unique(chunk1)
uid1 = uid1[uid1 > 0]  # omit bkg

# print('gid:',iou_gid)
print('uid:', np.shape(uid1))
print('iou:', np.shape(iou[iou_gid, 1]))

iou_rest_id = np.array(list(set(uid1)-set(iou[iou_gid,1])))# get rest ids
print(len(iou_rest_id))
print('1')
rl = np.zeros(chunk1.max()+1,int)
print('2')
rl[iou[iou_gid,1]] = iou[iou_gid,0]
print('3')
rl[iou_rest_id] = mid+np.arange(1,len(iou_rest_id)+1)
print('4')


chunk1 = rl[chunk1]


new_path = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Slice4096/for1200/n2seg_0.h5'
writeh5(new_path, 'main', chunk1)
print(np.shape(chunk1))

for j in range(200):
    ## save to png files for VAST
    label = chunk1[j]
    id = "%04d" % (j + i)
    png = seg2Vast(label)
    filedir = '/n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/Slice4096/for1200/ToVAST2/'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    file_path = filedir + str(id) + '_' + 'tr1-tc1.png'
    imwrite(file_path, png)

mid += len(iou_rest_id)  # for a chunk
print('mid:',mid)
print('done:', i)

