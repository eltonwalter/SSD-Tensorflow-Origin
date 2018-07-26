# Copyright 2016 Paul Balanca. All Rights Reserved.
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
"""Definition of 300 VGG-based SSD network.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
    with slim.arg_scope(ssd_vgg.ssd_vgg()):
        outputs, end_points = ssd_vgg.ssd_vgg(inputs)

This network port of the original Caffe model. The padding in TF and Caffe
is slightly different, and can lead to severe accuracy drop if not taken care
in a correct way!

In Caffe, the output size of convolution and pooling layers are computing as
following: h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1

Nevertheless, there is a subtle difference between both for stride > 1. In
the case of convolution:
    top_size = floor((bottom_size + 2*pad - kernel_size) / stride) + 1
whereas for pooling:
    top_size = ceil((bottom_size + 2*pad - kernel_size) / stride) + 1
Hence implicitely allowing some additional padding even if pad = 0. This
behaviour explains why pooling with stride and kernel of size 2 are behaving
the same way in TensorFlow and Caffe.

Nevertheless, this is not the case anymore for other kernel sizes, hence
motivating the use of special padding layer for controlling these side-effects.

@@ssd_vgg_300
"""
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common

slim = tf.contrib.slim


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=21,
        no_annotation_label=21,
        #得到feature map的层数，及feature map的shape
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        #对size的约束
        #论文中，S_k=S_min+(S_max-S_min)*(k-1)/(m-1),代码中为固定值21，45，99，153，207，261
        # anchor_size_bounds=[0.20, 0.90],
        
        #通过size和ritio可得到default boxes的W和H:
        #W=S_k*sqr(R),H=S_k/sqr(R)
        #加上W=H=sqr(S_k*S_(k+1))
        anchor_size_bounds=[0.15, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        #将feature map还原到原图（300*300）的比例
        anchor_steps=[8, 16, 32, 64, 100, 300],
        #定位到网格中心的偏置
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
        """SSD network definition.
        """
        #r[0]为predictions,r[1]为localisations
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    #预设网络公用的参数
    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def update_feature_shapes(self, predictions):
        """Update feature shapes from predictions collection (Tensor or Numpy
        array).
        """
        shapes = ssd_feat_shapes_from_net(predictions, self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)
    
    #找出所有anchors
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)
    #通过ssd_common对坐标进行编码
    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)
    #通过ssd_common对坐标进行解码
    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)
    
    #通过ssd_common选取合适的anchors
    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        #通过SSD网络，得到检测到的bbox
        
        # Select top_k bboxes from predictions, and clip
        #选取top_k=400个框，并对框做修建（超出原图尺寸范围的切掉） 
        #得到对应某个类别的得分值以及bbox  
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        #按照得分高低，筛选出400个bbox和对应得分  
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        #应用非极大值抑制，筛选掉与得分最高bbox重叠率大于0.5的，保留200个  
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes

#得到每个feature map中的点在原图中对应的坐标，和原图中其Scale对应的 DF_Box
def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      sizes: scale #Absolute reference sizes;
      ratios: Ratios to use on these features;
      step:从feature map还原到原图的比例
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    
    #生成网格的左上角坐标，与feature map shape相匹配（覆盖整个feature map）
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]   
    
    #加上偏移后得到网格的中心坐标，根据其得到default box映射到原图中的坐标，并将其归一化
    y = (y.astype(dtype) + offset) * step / img_shape[0]  
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    
    #每个feature map上default boxes的数量为 2+len(ratios)
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    
    #ratio=1时的两个default boxes
    #W=H=S_k
    #W=H=sqr(S_k*S_(k+1))
    h[0] = sizes[0] / img_shape[0]  #sizes[0]即这个feature map的scale
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    
    #ratio!=1时的default boxes
    #返回不同ratio对应的长和宽：W=S_k*sqr(R),H=S_k/sqr(R)
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    #返回所有网格的中心坐标在原图中对应的x和y，以及对应的原图中default boxes的W和H
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    #对feature map中的每个点，将其中心位置按比例映射到原图坐标
    #每个feature map对应一个Scale，以不同的ratio，在原图中设置DF_Box
    #得到每个feature map中的点在原图中对应的坐标，和原图中其Scale对应的 DF_Box
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        #将x的形状从(6，4)转化为list格式[]
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

#得到输入的网络对应的pediction和location
def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    #如果normalization > 0，则对input进行L2正则化
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    #每层特征图参考先验框的个数[4,6,6,6,4,4]
    num_anchors = len(sizes) + len(ratios)

    # Location.
    #每个框需要预测四个值
    num_loc_pred = num_anchors * 4
    #在原网络后加一个3*3的卷积层，kernel number = num_anchors * 4
    #卷积后的每个点为feat_layers每个cell的预测结果
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    #将数据格式调整为'NHWC'-[batch, in_height, in_width, in_channels]
    loc_pred = custom_layers.channel_to_last(loc_pred)
    #最后整个特征图所有锚点框预测目标位置 tensor为[h,w,每个cell先验框数，4]   [1,38,38,4,4] 
    #loc_pred.shape:(1, 38, 38, 16)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    #loc_pred.shape:(1, 38, 38, 4, 4)
    #print("loc_pred.shape2:",loc_pred.shape)
    
    # Class prediction.
    #每个框需要对每个类别进行预测
    num_cls_pred = num_anchors * num_classes
    #在原网络后加一个3*3的卷积层，kernel number = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    #最后整个特征图所有锚点框预测类别 tensor为[h,w*,每个cell先验框数，种类数]   [1,38,38,4,10]
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred

    #模型的网络结构
def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
    """SSD net definition.
    """
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        #之前为19*19，该max_pool2d padding=valid, 则经过maxpooling变为17*17
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        #i,layer分别为index和feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                #返回在每个feature map上对类别和位置的预测结果
                p, l = ssd_multibox_layer(end_points[layer],       #为feat_layers层之前的网络
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            #prediction_fn=slim.softmax，找到所有框的所有类别置信度
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
        return predictions, localisations, logits, end_points
    
ssd_net.default_image_size = 300

#预设网络公用的参数
def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Definese VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.

    Args:
      caffe_scope: Caffe scope object with loaded weights.

    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc


# =========================================================================== #
# SSD loss function.
#损失函数定义为位置误差和置信度误差的加权和；
#参数：每一个anchors预测的类别得分、预测的位置偏移量、真实类别、真实位置、真实得分
# =========================================================================== #
def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        #lshape=(1,38,38,4,21)
        lshape = tfe.get_shape(logits[0], 5)
        #num_classes=21
        num_classes = lshape[-1]
        #batch_size
        batch_size = lshape[0]

        #平齐所有向量
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        #len(logits)=6 feature map数量
        for i in range(len(logits)):
            #将预测的类别得分reshape成（-1,21）
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))  
            #真实的类别
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            #真实的类别得分
            fgscores.append(tf.reshape(gscores[i], [-1]))
            #将预测的边框坐标（编码形式的值） reshape成（-1,4） 
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            #用于将真实的边框坐标进行编码存储  
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        #将所有feature map的数据按行连接起来（叠放在列上），如flogits，从6*[-1,21]，变为[-1,21] （行数增多）
        #每一行数据包含一个anchor的信息：logits-预测的类别得分(21)，gclasses-真实的类别(1)，gscores-真实的类别得分(1)，localisations-预测的位置偏移(4)，glocalisations-真实的位置偏移(4)
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        # 正样本选择
        #如果anchors真实的类别得分>0.5则将这个预测框作为正样本 
        #pmask： shape=gscores_shape. if gscores>0.5, pmask=True;else pmask=False
        pmask = gscores > match_threshold
        #将pmask从True、False转化为1，0
        fpmask = tf.cast(pmask, dtype)
        #求 pmask=1的数量（作为正样本数量）
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        # Hard negative mining... 为了保证正负样本尽量平衡，SSD采用了hard negative mining，就是对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3  
        #no_classes：gscores>0.5, pmask=1 ；否则pmask=0
        # no_classes=1为正样本（正样本有物体，负样本为背景 ）
        no_classes = tf.cast(pmask, tf.int32)
        #predictions：预测为每个类别的概率
        predictions = slim.softmax(logits)
        #-0.5<gscore<0.5时nmask=True，代表负样本
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        #-0.5<gscore<0.5时fnmask=1，代表负样本
        fnmask = tf.cast(nmask, dtype)
        #tf.where(input, a,b)，其中a，b均为尺寸一致的tensor。
        #作用是将a中对应input中true的位置的元素值不变，其余元素进行替换，替换成b中对应位置的元素值
        #即当nmask=0时(gscores>0.5),将(正样本)对应的predictions[0]替换成1。将新的predictions[0]作为nvalues。
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        #将所有框的预测类别为背景的概率排成一行，保存在nvalues_flat中（其中正样本对应的概率为1）
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        #计算全部的负样本数量
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        #负样本数量是正样本数量的3倍+batch_size  
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        #如果全部的负样本数量小于n_neg，则n_neg等于全部的负样本数量
        n_neg = tf.minimum(n_neg, max_neg_entries)
        
        #抽样时按照置信度误差（预测为背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本
        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        #负样本背景置信度最大值（背景置信度小于该阈值为负样本）
        max_hard_pred = -val[-1]
        # Final negative mask.
        #-0.5<gscore<0.5,且nvalues<阈值，代表负样本
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        #负样本fnmask从True转化为1
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.#交叉熵
        #正样本误差：
        with tf.name_scope('cross_entropy_pos'):
            #类别置信度误差：已知对各个类别的预测得分和正确类别，通过学习让正确类别的得分尽可能大。  
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)
            #将正样本的置信度误差相加后除以batch-size  （通过fpmask对正样本进行筛选）
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)
        #负样本的误差
        with tf.name_scope('cross_entropy_neg'):
            #no_classes：背景为0，非背景为1
            #类别置信度误差：已知背景与非背景的预测得分和分类，通过学习让识别为背景的能力变强。
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            #将负样本（背景）的置信度误差相加后除以batch-size （通过fnmask对正样本进行筛选） 
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        #位置误差
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            #先验框对应边界的位置预测值-真实位置；然后做Smooth L1 loss  
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            #将正样本的loss*alpha，求和后除以batch-size （通过fpmask对正样本进行筛选）
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            #获得置信度误差和位置误差的加权和  
            tf.losses.add_loss(loss)


def ssd_losses_old(logits, localisations,
                   gclasses, glocalisations, gscores,
                   match_threshold=0.5,
                   negative_ratio=3.,
                   alpha=1.,
                   label_smoothing=0.,
                   device='/cpu:0',
                   scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.device(device):
        with tf.name_scope(scope, 'ssd_losses'):
            l_cross_pos = []
            l_cross_neg = []
            l_loc = []
            for i in range(len(logits)):
                dtype = logits[i].dtype
                with tf.name_scope('block_%i' % i):
                    # Sizing weight...
                    wsize = tf.get_shape(logits[i], rank=5)
                    wsize = wsize[1] * wsize[2] * wsize[3]

                    # Positive mask.
                    pmask = gscores[i] > match_threshold
                    fpmask = tf.cast(pmask, dtype)
                    n_positives = tf.reduce_sum(fpmask)

                    # Select some random negative entries.
                    # n_entries = np.prod(gclasses[i].get_shape().as_list())
                    # r_positive = n_positives / n_entries
                    # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                    # Negative mask.
                    no_classes = tf.cast(pmask, tf.int32)
                    predictions = slim.softmax(logits[i])
                    nmask = tf.logical_and(tf.logical_not(pmask),
                                           gscores[i] > -0.5)
                    fnmask = tf.cast(nmask, dtype)
                    nvalues = tf.where(nmask,
                                       predictions[:, :, :, :, 0],
                                       1. - fnmask)
                    nvalues_flat = tf.reshape(nvalues, [-1])
                    # Number of negative entries to select.
                    n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                    n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                    n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                    max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                    n_neg = tf.minimum(n_neg, max_neg_entries)

                    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                    max_hard_pred = -val[-1]
                    # Final negative mask.
                    nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
                    fnmask = tf.cast(nmask, dtype)

                    # Add cross-entropy loss.
                    with tf.name_scope('cross_entropy_pos'):
                        fpmask = wsize * fpmask
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                              labels=gclasses[i])
                        loss = tf.losses.compute_weighted_loss(loss, fpmask)
                        l_cross_pos.append(loss)

                    with tf.name_scope('cross_entropy_neg'):
                        fnmask = wsize * fnmask
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                              labels=no_classes)
                        loss = tf.losses.compute_weighted_loss(loss, fnmask)
                        l_cross_neg.append(loss)

                    # Add localization loss: smooth L1, L2, ...
                    with tf.name_scope('localization'):
                        # Weights Tensor: positive mask + random negative.
                        weights = tf.expand_dims(alpha * fpmask, axis=-1)
                        loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                        loss = tf.losses.compute_weighted_loss(loss, weights)
                        l_loc.append(loss)

            # Additional total losses...
            with tf.name_scope('total'):
                total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
                total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
                total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
                total_loc = tf.add_n(l_loc, 'localization')

                # Add to EXTRA LOSSES TF.collection
                tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
                tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
                tf.add_to_collection('EXTRA_LOSSES', total_cross)
                tf.add_to_collection('EXTRA_LOSSES', total_loc)
