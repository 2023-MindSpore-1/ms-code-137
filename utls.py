import math
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops.function.array_func import nonzero_

sort = ops.Sort(axis=-1)
sort_0 = ops.Sort(axis=0)


def transform_numpy(ipt):
    if type(ipt) is not np.ndarray:
        ipt = np.array(ipt)

    return ipt

def ms_argwhere(x):
    return nonzero_(x)


def compute_ap_cmc(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap = ap + d_recall * precision

    return ap, cmc


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    num_q, num_g = distmat.shape
    _, index = sort(distmat.astype(ms.float32))  # from small to large

    num_no_gt = 0  # num of query imgs without groundtruth
    num_r1 = 0
    # CMC = ms.Tensor(np.zeros(g_pids.shape[0]), ms.float32)
    CMC = np.zeros(len(g_pids))
    AP = 0

    # q_pids, g_pids, q_camids, g_camids = q_pids.asnumpy(), g_pids.asnumpy(), q_camids.asnumpy(), g_camids.asnumpy()
    for i in range(num_q):
        # groundtruth index
        query_index = ms_argwhere(g_pids == q_pids[i])
        camera_index = ms_argwhere(g_camids == q_camids[i])
        good_index = np.setdiff1d(query_index.asnumpy(), camera_index.asnumpy(),
                                  assume_unique=True)  # 求两个数组的集合差。返回' ar1 '中不在' ar2 '中的唯一值。
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index.asnumpy(), camera_index.asnumpy())  # 常见和唯一元素的排序一维数组。
        index_tmp = index[i].asnumpy()
        ap_tmp, CMC_tmp = compute_ap_cmc(index_tmp, good_index, junk_index)
        if CMC_tmp[0] == 1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query imgs do not have groundtruth.".format(num_no_gt))

    # print("R1:{}".format(num_r1))

    CMC = CMC / (num_q - num_no_gt + 1e-6)
    mAP = AP / (num_q - num_no_gt + 1e-6)

    return CMC, mAP


reshape = ops.Reshape()


def HPA(R, topK):
    galleryNum = R.shape[0]
    rankerNum = R.shape[1]
    averageRank = R.sum(axis=1)
    _, pseudoRank1 = sort(averageRank.astype(ms.float32))
    _, pseudoRank = sort(pseudoRank1.astype(ms.float32))
    _, R1 = sort_0(R.astype(ms.float32))

    NDCG = ms.Tensor(np.zeros((rankerNum, 1), np.float32), ms.float32)
    for i in range(rankerNum):
        for j in range(topK):
            if R1[j, i] == pseudoRank1[j]:
                ri = 1
            else:
                ri = 0
            NDCG[i] = NDCG[i] + ri * math.log(2) / math.log(i + 2)
    _, NDCGrank = sort(-NDCG)
    finalRank = np.zeros((galleryNum, 1), np.float32)
    finalRank = ms.Tensor(finalRank, ms.float32)
    for i in range(rankerNum):
        tmp = NDCG[i, 0] * R[:, i]
        finalRank = reshape(tmp, (1000, 1)) + finalRank

    _, finalRank = sort_0(finalRank)
    _, finalRank = sort_0(finalRank.astype(ms.float32))
    return finalRank