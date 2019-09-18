import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from model.config import cfg

a = np.random.random((1,800,600,3))
b = np.array([800,600,3])
c = np.array([[100,200,500,400,1]])

image_placeholder = tf.placeholder(tf.float32,[1,None,None,3])
imgsize_placeholder = tf.placeholder(tf.float32,[3])
label_placeholder = tf.placeholder(tf.float32,[None,5])

mode = 'TRAIN'
# mode = 'TEST'


_predictions = {}
_losses = {}
_anchor_targets = {}
_proposal_targets = {}
_layers = {}
_gt_image = None
_act_summaries = []
_score_summaries = {}
_train_summaries = []
_event_summaries = {}
_variables_to_fix = {}

num_classes = 2
anchor_scales=(8, 16, 32)
anchor_ratios=(0.5, 1, 2)

num_scales = len(anchor_scales)
num_ratios = len(anchor_ratios)
num_anchors = num_scales * num_ratios
feat_stride = [16, ]

initializer = tf.contrib.layers.xavier_initializer_conv2d()

def conv_op(input_op, name, kh, kw, n_out, dh, dw,padding):
    # input_op = tf.convert_to_tensor(input_op)
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                shape = [kh, kw, n_in, n_out],
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=tf.contrib.layers.l2_regularizer(0.001))
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding)
        bias_init_val = tf.constant(0.0, shape = [n_out], dtype = tf.float32)
        biases = tf.Variable(bias_init_val, trainable = True, name = 'b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name = scope)
        return activation
 

def fc_op(input_op, name, n_out):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                shape = [n_in, n_out],
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(0.001))
        biases = tf.Variable(tf.constant(0.1, shape = [n_out], dtype = tf.float32), name = 'b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope) 
        return activation
    

def mpool_op(input_op, name, kh, kw, dh, dw):
    return  tf.nn.max_pool(input_op,
                           ksize = [1, kh, kw, 1],
                           strides = [1, dh, dw, 1],
                           padding = 'SAME',
                           name = name)

def _reshape_layer(bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
                # change the channel to the caffe format
                to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
                # then force it to have channel 2
                reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
                # then swap the channel back
                to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
                return to_tf

def _softmax_layer(bottom, name):

        if name.startswith('rpn_cls_prob_reshape'):
                input_shape = tf.shape(bottom)
                bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
                reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
                return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)


def vgg_head(input_op, keep_prob):
    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, padding = 'SAME')
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, padding = 'SAME')
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)
 
    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, padding = 'SAME')
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, padding = 'SAME')
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)
 
    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, padding = 'SAME')
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, padding = 'SAME')
    conv3_3 = conv_op(conv3_2,  name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, padding = 'SAME')    
    pool3 = mpool_op(conv3_3,   name="pool3",   kh=2, kw=2, dh=2, dw=2)
 
    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, padding = 'SAME')
    conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, padding = 'SAME')
    conv4_3 = conv_op(conv4_2,  name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, padding = 'SAME')
    pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)
 
    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, padding = 'SAME')
    conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, padding = 'SAME')
    conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, padding = 'SAME')
    
 

    return conv5_3

def vgg16_head(input, is_training):

        with tf.variable_scope('vgg_16') as scope:
                net = slim.repeat(input, 2, slim.conv2d, 64, [3, 3],trainable=True, scope='conv1')

                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],trainable=False, scope='conv2')

                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],trainable=is_training, scope='conv3')

                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],trainable=is_training, scope='conv4')

                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],trainable=is_training, scope='conv5')

        _act_summaries.append(net)
        _layers['head'] = net
    
        return net

def vgg16_tail(pool5, is_training):
        with tf.variable_scope('vgg_tail') as scope:
    
                pool5_flat = slim.flatten(pool5, scope='flatten')
                fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
                if is_training:
                        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, 
                                scope='dropout6')
                fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
                if is_training:
                        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, 
                                scope='dropout7')

        return fc7

#rpn model
def _region_proposal(net,is_training):


        # rpn  = conv_op(net,  name="rpn_conv3x3", kh=3, kw=3, n_out=512, dh=1, dw=1, padding='SAME')
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training,scope="rpn_conv/3x3")
        _act_summaries.append(rpn)
        # rpn_cls_score = conv_op(rpn, name="rpn_cls_score", kh=3, kw=3, n_out=num_anchors*2, dh=1, dw=1, padding = 'SAME')#1x48x36x18
        rpn_cls_score = slim.conv2d(rpn, num_anchors * 2, [1, 1], trainable=is_training,padding='SAME', activation_fn=None, scope='rpn_cls_score')
        rpn_cls_score_reshape = _reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')#1x432x36x2
        rpn_cls_prob_reshape = _softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")#1x432x36x2
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")#15552
        rpn_cls_prob = _reshape_layer(rpn_cls_prob_reshape, num_anchors * 2, "rpn_cls_prob")#1x48x36x18

        # rpn_bbox_pred = conv_op(rpn,  name="rpn_bbox_pred", kh=3, kw=3, n_out=num_anchors*4, dh=1, dw=1, padding='SAME')#1x48x36x36
        rpn_bbox_pred = slim.conv2d(rpn, num_anchors * 4, [1, 1], trainable=is_training,padding='SAME', activation_fn=None, scope='rpn_bbox_pred')

        if is_training:

                rois, roi_scores = _proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                rpn_labels = _anchor_target_layer(rpn_cls_score, "anchor")
                # Try to have a deterministic order for the computing graph, for reproducibility
                with tf.control_dependencies([rpn_labels]):
                        rois, _ = _proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
                if cfg.TEST.MODE == 'nms':
                        rois, _ = _proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                elif cfg.TEST.MODE == 'top':
                        rois, _ = _proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                else:
                        raise NotImplementedError
        
        _predictions["rpn_cls_score"] = rpn_cls_score
        _predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        _predictions["rpn_cls_prob"] = rpn_cls_prob
        _predictions["rpn_cls_pred"] = rpn_cls_pred
        _predictions["rpn_bbox_pred"] = rpn_bbox_pred
        _predictions["rois"] = rois
        return rois
        # return rpn_labels
        

def _proposal_target_layer(rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
                rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                        proposal_target_layer,
                        [rois, roi_scores, label_placeholder, num_classes],
                        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                        name="proposal_target")

                rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
                roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
                labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
                bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, num_classes * 4])
                bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, num_classes * 4])
                bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, num_classes * 4])

                _proposal_targets['rois'] = rois
                _proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
                _proposal_targets['bbox_targets'] = bbox_targets
                _proposal_targets['bbox_inside_weights'] = bbox_inside_weights
                _proposal_targets['bbox_outside_weights'] = bbox_outside_weights

                _score_summaries.update(_proposal_targets)

                return rois, roi_scores

def _proposal_layer(rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:

                rois, rpn_scores = proposal_layer_tf(
                        rpn_cls_prob,
                        rpn_bbox_pred,
                        imgsize_placeholder,
                        mode,
                        feat_stride,
                        anchors,
                        num_anchors
                )
      

        rois.set_shape([None, 5])
        rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

def _anchor_target_layer(rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, label_placeholder, imgsize_placeholder, feat_stride, anchors, num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

        rpn_labels.set_shape([1, 1, None, None])
        rpn_bbox_targets.set_shape([1, None, None, num_anchors * 4])
        rpn_bbox_inside_weights.set_shape([1, None, None, num_anchors * 4])
        rpn_bbox_outside_weights.set_shape([1, None, None, num_anchors * 4])

        rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
        _anchor_targets['rpn_labels'] = rpn_labels
        _anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
        _anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        _anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        _score_summaries.update(_anchor_targets)

        return rpn_labels

def _crop_pool_layer(bottom, rois, name):
        with tf.variable_scope(name) as scope:
                batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
                # Get the normalized coordinates of bounding boxes
                bottom_shape = tf.shape(bottom)
                height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(feat_stride[0])
                width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(feat_stride[0])
                x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
                y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
                x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
                y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
                # Won't be back-propagated to rois anyway, but to save time
                bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
                #128x14x14x512
        return mpool_op(crops,   name="pool5",   kh=2, kw=2, dw=2, dh=2)

def _region_classification(fc7, is_training, initializer, initializer_bbox):
        cls_score = slim.fully_connected(fc7, num_classes, weights_initializer=initializer,trainable=is_training,activation_fn=None, scope='cls_score')
        cls_prob = _softmax_layer(cls_score, "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        bbox_pred = slim.fully_connected(fc7, num_classes * 4, weights_initializer=initializer_bbox,trainable=is_training,activation_fn=None, scope='bbox_pred')

        _predictions["cls_score"] = cls_score
        _predictions["cls_pred"] = cls_pred
        _predictions["cls_prob"] = cls_prob
        _predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred   

def network(is_training=True):
        global anchors
        with tf.variable_scope('network') as scope:
                net = vgg16_head(image_placeholder,1.0)



                # just to get the shape right
                height = tf.to_int32(tf.ceil(imgsize_placeholder[0] / np.float32(16)))
                width = tf.to_int32(tf.ceil(imgsize_placeholder[1] / np.float32(16)))

                anchors, anchor_length = generate_anchors_pre_tf(height,width)
        
                anchors.set_shape([None, 4])
                anchor_length.set_shape([])

                rois = _region_proposal(net,is_training)

                if cfg.POOLING_MODE == 'crop':
                        pool5 = _crop_pool_layer(net, rois, "pool5")#128x7x7x512
                else:
                        raise NotImplementedError

                fc7 = vgg16_tail(pool5, is_training)

                cls_prob, bbox_pred = _region_classification(fc7, is_training, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01), initializer_bbox=tf.truncated_normal_initializer(mean=0.0, stddev=0.001))
        _score_summaries.update(_predictions)
        return rois, cls_prob, bbox_pred

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box,axis=dim))
        return loss_box

rois, cls_prob, bbox_pred=network(True)

with tf.variable_scope('LOSS') as scope:

        rpn_cls_score = tf.reshape(_predictions['rpn_cls_score_reshape'], [-1, 2])
        rpn_label = tf.reshape(_anchor_targets['rpn_labels'], [-1])
        rpn_select = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # RPN, bbox loss
        rpn_bbox_pred = _predictions['rpn_bbox_pred']
        rpn_bbox_targets = _anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = _anchor_targets['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = _anchor_targets['rpn_bbox_outside_weights']
        rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,rpn_bbox_outside_weights, sigma=3.0, dim=[1, 2, 3])

        # RCNN, class loss
        cls_score = _predictions["cls_score"]
        label = tf.reshape(_proposal_targets["labels"], [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # RCNN, bbox loss
        bbox_pred = _predictions['bbox_pred']
        bbox_targets = _proposal_targets['bbox_targets']
        bbox_inside_weights = _proposal_targets['bbox_inside_weights']
        bbox_outside_weights = _proposal_targets['bbox_outside_weights']
        loss_box = _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        _losses['cross_entropy'] = cross_entropy
        _losses['loss_box'] = loss_box
        _losses['rpn_cross_entropy'] = rpn_cross_entropy
        _losses['rpn_loss_box'] = rpn_loss_box

        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        # regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
        # _losses['total_loss'] = loss + regularization_loss
        #暂未加l2正则


        _event_summaries.update(_losses)
       



with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(),tf.local_variables_initializer()))

        out = sess.run(loss,feed_dict={image_placeholder:a,imgsize_placeholder:b,label_placeholder:c})
        print(out)
        print(out.shape)