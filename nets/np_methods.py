# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Additional Numpy methods. Big mess of many things!
"""
import numpy as np


# =========================================================================== #
# Numpy implementations of SSD boxes functions.
# =========================================================================== #
#将预测的位置坐标解码成预测框的真实坐标
def ssd_bboxes_decode(feat_localizations,
                      anchor_bboxes,
                      prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Return:
      numpy array Nx4: ymin, xmin, ymax, xmax
    """
    # Reshape for easier broadcasting.
    l_shape = feat_localizations.shape
    feat_localizations = np.reshape(feat_localizations,
                                    (-1, l_shape[-2], l_shape[-1]))
    yref, xref, href, wref = anchor_bboxes
    xref = np.reshape(xref, [-1, 1])
    yref = np.reshape(yref, [-1, 1])

    # Compute center, height and width
    cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, 1] * href * prior_scaling[1] + yref
    w = wref * np.exp(feat_localizations[:, :, 2] * prior_scaling[2])
    h = href * np.exp(feat_localizations[:, :, 3] * prior_scaling[3])
    # bboxes: ymin, xmin, xmax, ymax.
    bboxes = np.zeros_like(feat_localizations)
    bboxes[:, :, 0] = cy - h / 2.
    bboxes[:, :, 1] = cx - w / 2.
    bboxes[:, :, 2] = cy + h / 2.
    bboxes[:, :, 3] = cx + w / 2.
    # Back to original shape.
    bboxes = np.reshape(bboxes, l_shape)
    return bboxes

#根据阈值对预测框进行筛选
def ssd_bboxes_select_layer(predictions_layer,
                            localizations_layer,
                            anchors_layer,
                            select_threshold=0.5,
                            img_shape=(300, 300),
                            num_classes=21,
                            decode=True):
    """Extract classes, scores and bounding boxes from features in one layer.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    # First decode localizations features if necessary.
    #根据位置偏移量和anchor的位置进行解码，得到对应的真实位置坐标[Y,X,H,W]
    if decode:
        localizations_layer = ssd_bboxes_decode(localizations_layer, anchors_layer)

    # Reshape features to: Batches x N x N_labels | 4.
    #p_shape:[1,18,18,4,21]
    p_shape = predictions_layer.shape
    #len(p_shape)=5,layer1-batch_size=p_shape[0]=1
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    #reshape:[1,-1,21]
    predictions_layer = np.reshape(predictions_layer,
                                   (batch_size, -1, p_shape[-1]))
    #l_shape:[1,18,18,4,4]
    l_shape = localizations_layer.shape
    ##reshape:[1,-1,4]
    localizations_layer = np.reshape(localizations_layer,
                                     (batch_size, -1, l_shape[-1]))

    # Boxes selection: use threshold or score > no-label criteria.
    if select_threshold is None or select_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        #找到predictions_layer第三列最大的值并返回其索引,返回shape=[1,-1,1]
        classes = np.argmax(predictions_layer, axis=2)
        #找到predictions_layer第三列最大的值并返回该值,返回shape=[1,-1,1]
        scores = np.amax(predictions_layer, axis=2)
        #classes=0时mask=False，classes>0时mask=True
        mask = (classes > 0)
        #将类别判断为背景的数据（predictions和localisations）筛选掉，保留剩余的数据
        classes = classes[mask]
        scores = scores[mask]
        bboxes = localizations_layer[mask]
    else:
        #去除掉背景类别
        sub_predictions = predictions_layer[:, :, 1:]
        #将类别置信度大于select_threshold的数据筛选出来，idxes=[索引1，索引2，索引3] 根据索引1和索引2找到每条数据，索引3代表类别
        idxes = np.where(sub_predictions > select_threshold)
        #idxes[-1]代表每个类别置信度大于阈值的类别列索引
        #去掉背景一列后序号发生改变，需要+1恢复原来的索引
        classes = idxes[-1]+1
        #scores为类别置信度大于阈值的类别得分
        scores = sub_predictions[idxes]
        #idxes[:-1]代表每个大于阈值的数据索引
        #bboxes为类别置信度大于阈值的数据对应的位置坐标[Y,X,H,W]
        bboxes = localizations_layer[idxes[:-1]]

    return classes, scores, bboxes

#选择大于阈值的预测框
def ssd_bboxes_select(predictions_net,
                      localizations_net,
                      anchors_net,
                      select_threshold=0.5,
                      img_shape=(300, 300),
                      num_classes=21,
                      decode=True):
    """Extract classes, scores and bounding boxes from network output layers.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    # l_layers = []
    # l_idxes = []
    for i in range(len(predictions_net)):
        classes, scores, bboxes = ssd_bboxes_select_layer(
            predictions_net[i], localizations_net[i], anchors_net[i],
            select_threshold, img_shape, num_classes, decode)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)
        # Debug information.
        # l_layers.append(i)
        # l_idxes.append((i, idxes))
    #通过np.concatenate将list中的数据连接起来
    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes


# =========================================================================== #
# Common functions for bboxes handling and selection.
# =========================================================================== #
#选择得分top_k的数据
def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes

#将大于图像大小的框裁剪到图像大小
def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes

#根据图像大小对预测框的数值进行归一化，假设图像大小：[0, 0, 1, 1]
def bboxes_resize(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform.
    """
    bboxes = np.copy(bboxes)
    # Translate.
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    # Resize.
    resize = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= resize[0]
    bboxes[:, 1] /= resize[1]
    bboxes[:, 2] /= resize[0]
    bboxes[:, 3] /= resize[1]
    return bboxes

#计算两个预测框的IOU重叠度
def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    #计算出重叠部分的四个坐标
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    #计算出重叠部分的面积
    int_vol = int_h * int_w
    # Union volume.
    #计算两个框的面积
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    #返回IOU
    return jaccard


def bboxes_intersection(bboxes_ref, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes_ref = np.transpose(bboxes_ref)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes_ref[0], bboxes2[0])
    int_xmin = np.maximum(bboxes_ref[1], bboxes2[1])
    int_ymax = np.minimum(bboxes_ref[2], bboxes2[2])
    int_xmax = np.minimum(bboxes_ref[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol = (bboxes_ref[2] - bboxes_ref[0]) * (bboxes_ref[3] - bboxes_ref[1])
    score = int_vol / vol
    return score

#非极大化抑制算法
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    #如果keep_bboxes=0代表删去该预测框
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        #仅对保留下来的预测框进行判断
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            #overlap < nms_threshold或类别不同时keep_overlap=1，否则=0
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            #与操作：一个为0则结果为0.（keep_bboxes与keep_overlap）
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)
    #keep_bboxes=0，idxes=False；keep_bboxes=1，idxes=True；
    idxes = np.where(keep_bboxes)
    #保留idxes=True的数据
    return classes[idxes], scores[idxes], bboxes[idxes]


def bboxes_nms_fast(classes, scores, bboxes, threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    pass




