from __future__ import print_function, absolute_import
import numpy as np
# import h5py
import scipy.io as sio
from mindspore import ops
import mindspore as ms
from utls import evaluate, HPA, transform_numpy

# feature = h5py.File('/opt/data/private/shaohua/dataset/BMVC22/dataset/dukemtmcreid/ranklist/test/test.mat')
# sim = feature['Test'][:]
sim = sio.loadmat('./dataset/ranklist/test/test.mat')
gallery_label = sio.loadmat('./dataset/dukemtmcreid/groundtruth/galleryID.mat')
query_label = sio.loadmat('./dataset/dukemtmcreid/groundtruth/queryIDtest.mat')
cam_gallery = sio.loadmat('./dataset/dukemtmcreid/groundtruth/galleryCAM.mat')
cam_query1 = sio.loadmat('./dataset/dukemtmcreid/groundtruth/queryCAM.mat')
cam_query = cam_query1[1685:3368]

# ---------------------------------------------------------
# sim = np.load("sim.npy")
# sim = transform_numpy(sim)
# query_label = [1 for i in range(1400)]
# query_label = transform_numpy(query_label)
# query_label = query_label[701:1400]
# gallery_label = [1 for i in range(1000)]
# gallery_label = transform_numpy(gallery_label)
# cam_gallery = [1 for i in range(1000)]
# cam_gallery = transform_numpy(cam_gallery)
# cam_query1 = [2 for i in range(1400)]
# cam_query1 = transform_numpy(cam_query1)
# cam_query = cam_query1[701:1400]
# query_label = ms.Tensor(query_label, dtype=ms.float32)
# gallery_label = ms.Tensor(gallery_label, dtype=ms.float32)
# cam_gallery = ms.Tensor(cam_gallery, dtype=ms.float32)
# cam_query = ms.Tensor(cam_query, dtype=ms.float32)
# ---------------------------------------------------------
rankernum = sim.shape[0]
querynum = sim.shape[1]
gallerynum = sim.shape[2]
sim = ms.Tensor(sim, dtype=ms.float32)
sort = ops.Sort(axis=-1)
_, ranklist = sort(-sim)
_, rank = sort(ranklist.astype(ms.float32))
topK = 10
result = np.zeros((querynum, gallerynum), np.float32)
result = ms.Tensor(result, ms.float32)
for i in range(querynum):
    finalRank = HPA(rank[:, i, :].transpose(), topK)
    result[i, :] = finalRank

# eval
CMC_result, map_result = evaluate(result, query_label, gallery_label, cam_query, cam_gallery)
auc_result = 0.5 * (2 * np.sum(CMC_result) - CMC_result[0] - CMC_result[-1]) / (len(CMC_result) - 1);
result_str = 'Rank1 = %.5f | Rank5 = %.5f| Rank10 = %.5f| Rank20 = %.5f| auc = %.5f| map = %.5f\n' % (
        CMC_result[0], CMC_result[4],CMC_result[9],CMC_result[19],auc_result,map_result)
print(result_str)
with open('./results.txt', 'a') as file:
    file.write(result_str)