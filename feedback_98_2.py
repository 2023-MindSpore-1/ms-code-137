import math
import scipy.io as sio
import numpy as np
from mindspore import ops
import mindspore as ms
from utls import evaluate, ms_argwhere, transform_numpy

# sim = np.random.normal(size=(4, 100, 1000))
# sim = transform_numpy(sim)
# np.save("sim.npy",sim)
# sim = np.load("sim.npy")
sim = sio.loadmat('./test.mat')
# sim = np.load("sim.npy")
rankernum = sim.shape[0]
querynum = sim.shape[1]
gallerynum = sim.shape[2]

query_label = sio.loadmat('./dataset/cuhk03labeled-query_id-.mat')
query_label = query_label[701:1400]
gallery_label = sio.loadmat('./dataset/cuhk03labeled-gallery_idtest-.mat')
cam_gallery = sio.loadmat('./dataset/cuhk03labeled-gallery_camidstest-.mat')
cam_query1 = sio.loadmat('./dataset/cuhk03labeled-query_camids-.mat')
cam_query = cam_query1[701:1400]

#
# query_label = [1 for i in range(1400)]
# query_label = transform_numpy(query_label)
# query_label = query_label[701:1400]
#
# gallery_label = [1 for i in range(1000)]
# gallery_label = transform_numpy(gallery_label)
#
# cam_gallery = [1 for i in range(1000)]
# cam_gallery = transform_numpy(cam_gallery)
#
# cam_query1 = [2 for i in range(1400)]
# cam_query1 = transform_numpy(cam_query1)
# cam_query = cam_query1[701:1400]

sim = ms.Tensor(sim, dtype=ms.float32)
query_label = ms.Tensor(query_label, dtype=ms.float32)
gallery_label = ms.Tensor(gallery_label, dtype=ms.float32)
cam_gallery = ms.Tensor(cam_gallery, dtype=ms.float32)
cam_query = ms.Tensor(cam_query, dtype=ms.float32)

sort = ops.Sort(axis=-1)


_, ranklist = sort(-sim)
_, rank = sort(ranklist.astype(ms.float32))

averageRank = sim.sum(axis=0)

# averageRank = np.reshape(averageRank, (querynum, gallerynum))
_, pseudoRanklist = sort(-averageRank)
_, pseudoRank = sort(pseudoRanklist.astype(ms.float32))

# # 设定参数
lamda = 0.1
iteration = 10
HPA = 0
topK = 2 # 10

ops_max = ops.ReduceMax(keep_dims=False)
# 2.25
if HPA:
    NDCG = np.zeros((querynum, rankernum), np.float32)
    NDCG = ms.Tensor(NDCG, ms.float32)
    for i in range(querynum):
        for j in range(rankernum):
            for k in range(topK):
                if ranklist[j, i, k] in pseudoRanklist[i, :]:
                    ri = 1
                else:
                    ri = 0
                NDCG[i, j] = NDCG[i, j] + ri * math.log(2) / (math.log(k + 1) + 1e-6)
        maxndcg = ops_max(NDCG[i, :], axis=0)
        NDCG[i, :] = NDCG[i, :] / maxndcg
    weight = NDCG
else:
    weight = np.ones((querynum, rankernum))
    weight = ms.Tensor(weight, ms.float32)

# get origin rank

origin_sim = ms.Tensor(np.zeros((querynum, gallerynum)), ms.float32)
sim = ms.Tensor(sim, ms.float32)
for i in range(querynum):
    for j in range(rankernum):
        origin_sim[i, :] = origin_sim[i, :] + (sim[j, i, :] * weight[i, j])

_, origin_ranklist = sort(-origin_sim)
_, origin_rank = sort(origin_ranklist.astype(ms.float32))
total_ranklist = origin_ranklist

result_1 = []
CMC_result, map_result = evaluate(origin_rank, query_label, gallery_label, cam_query, cam_gallery)
auc_result = 0.5 * (2 * sum(CMC_result) - CMC_result[0] - CMC_result[-1]) / (len(CMC_result) - 1)
result_1 = [CMC_result[0] * 100, auc_result, map_result]
print('original r1: %.2f mAP: %.2f \n' % (100 * CMC_result[0], 100 * map_result))


feedtrue_G = ms.Tensor(np.zeros((querynum, gallerynum)), ms.float32)
feeded_G = ms.Tensor(np.zeros((querynum, gallerynum)), ms.float32)


for i in range(iteration):
    new_weight = ms.Tensor(np.zeros((querynum, rankernum)), ms.float32)
    # feature_ranklist = ranklist[:, :, 1:(topK * i)]
    for q in range(querynum):
        Qlabel = query_label[q]

        RT = []
        sed = 0
        now_num = 1
        while sed < topK:
            if feeded_G[q, total_ranklist[q, now_num].asnumpy().item()] == 0:
                sed = sed + 1
                # RT(sed) = total_ranklist(q,now_num)
                RT.append(total_ranklist[q, now_num].asnumpy().item())
                feeded_G[q, total_ranklist[q, now_num].asnumpy().item()] = 1
            now_num = now_num + 1

        RT_label = gallery_label[RT]
        scored_G = ms_argwhere(RT_label == Qlabel)
        for j in range(topK):
            if j in scored_G.asnumpy():
                feedtrue_G[q, RT[j]] = 10
            else:
                feedtrue_G[q, RT[j]] = -10

        # scored_G = np.argwhere(feedtrue_G[q,:]==10)
        scored_G = ms_argwhere(feedtrue_G[q, :] == 10)

        if scored_G.shape[1] > 1:
            anno_G = sim[:, q, scored_G.asnumpy()]
            # 若 ms.ops.std 支持GPU后可进行修改
            std_w = np.std(anno_G, axis=1)
            max_std = np.max(std_w)
            std_w = std_w / max_std
            new_weight[q, :] = new_weight[q, :] + np.reshape(1 / std_w, rankernum)
            total_weight = np.max(new_weight[q, :])
            new_weight[q, :] = new_weight[q, :] / total_weight
    weight = weight * lamda + new_weight * (1 - lamda)
    for j in range(querynum):
        weight[j, :] = weight[j, :] / ops_max(weight[j, :])

    new_sim = ms.Tensor(np.zeros((querynum, gallerynum)), ms.float32)
    for j in range(querynum):
        for k in range(rankernum):
            new_sim[j, :] = new_sim[j, :] + (sim[k, j, :] * weight[j, k])

    new_sim = new_sim + feedtrue_G
    _,total_ranklist = sort(-new_sim)
    _,total_rank = sort(total_ranklist.astype(ms.float32))
    CMC_result, map_result = evaluate(total_rank, query_label, gallery_label, cam_query, cam_gallery)
    auc_result = 0.5 * (2 * np.sum(CMC_result) - CMC_result[0] - CMC_result[-1]) / (len(CMC_result) - 1)
    result_1 = [result_1, CMC_result[0] * 100, auc_result, map_result]
    print('iteration:%d r1:%.2f mAP:%.2f \n' % (i + 1, 100 * CMC_result[0], 100 * map_result))