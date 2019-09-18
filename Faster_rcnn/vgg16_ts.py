import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os

from tensorflow.python import pywrap_tensorflow
from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from model.config import cfg
import cv2

# np.set_printoptions(threshold=np.inf)
# a = np.random.random((1,800,600,3))
b = np.array([600,800,3])
c = np.array([[99,102,656,659,1],[99,102,656,659,1],[99,102,656,659,1],[99,102,656,659,1],[99,102,656,659,1]])
img = cv2.imread('G:\\AFLW\\800_600_images\\3\\image00035.jpg',1)
a = img[np.newaxis,:,:,:]



# train_path = '/home/uwe-1/Documents/Faster_rcnn_dataset/faster_rcnn_train.tfrecords'
# test_path = '/home/uwe-1/Documents/Faster_rcnn_dataset/faster_rcnn_test.tfrecords'
# train_path = 'G:\\AFLW\\800_600_images\\faster_rcnn_train.tfrecords'
# test_path = 'G:\\AFLW\\800_600_images\\faster_rcnn_test.tfrecords'

# def read_and_decode(file_path):
#     files = tf.train.match_filenames_once(file_path)
#     filename_queue = tf.train.string_input_producer(files,num_epochs=None,shuffle=False)

#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)

#     features = tf.parse_single_example(
#         serialized_example,
#         features={
#             'images':tf.FixedLenFeature([],tf.string),
#             'head_pose':tf.FixedLenFeature([],tf.string),
#             'box_label':tf.FixedLenFeature([],tf.string),
#             'image_size':tf.FixedLenFeature([],tf.string)
#         }
#     )
#     image = tf.decode_raw(features['images'],tf.uint8)
#     #image.set_shape([128,128,4])
#     image = tf.reshape(image,[800,600,3])
#     image = tf.cast(image,tf.float32)
# #     image = tf.image.per_image_standardization(image)

#     head_pose = tf.decode_raw(features['head_pose'],tf.float64)
#     head_pose = tf.reshape(head_pose,[3]) 
#     head_pose = tf.cast(head_pose,tf.float32)

#     box_label = tf.decode_raw(features['box_label'],tf.float64)
#     box_label = tf.reshape(box_label,[1,5]) 
#     box_label = tf.cast(box_label,tf.float32)

#     image_size = tf.decode_raw(features['image_size'],tf.float64)
#     image_size = tf.reshape(image_size,[3]) 
#     image_size = tf.cast(image_size,tf.float32)


#     return image,head_pose,box_label,image_size

# image_train,head_pose_train,box_label_train,image_size_train = read_and_decode(train_path)
# # min_after_dequeue = 1000
# # batch_size = 1
# # capacity = min_after_dequeue + 3 * batch_size

# # image_train_batch,head_pose_train_batch,box_label_train_batch,image_size_train_batch = tf.train.shuffle_batch(
# #     [image_train,head_pose_train,box_label_train,image_size_train],batch_size = batch_size, capacity = capacity, min_after_dequeue = min_after_dequeue,num_threads=2
# # )


# image_test,head_pose_test,box_label_test,image_size_test = read_and_decode(test_path)
# # image_test_batch,head_pose_test_batch,box_label_test_batch,image_size_test_batch = tf.train.shuffle_batch(
# #     [image_test,head_pose_test,box_label_test,image_size_test],batch_size = batch_size, capacity = capacity, min_after_dequeue = min_after_dequeue,num_threads=2
# # )

# # box_label_train_batch = tf.squeeze(box_label_train_batch,axis=0)
# # image_size_train_batch = tf.squeeze(image_size_train_batch,axis=0)
# # box_label_test_batch = tf.squeeze(box_label_test_batch,axis=0)
# # image_size_test_batch = tf.squeeze(image_size_test_batch,axis=0)
# image_train = tf.expand_dims(image_train,axis=0)
# image_test = tf.expand_dims(image_test,axis=0)

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
    
        pool5_flat = slim.flatten(pool5, scope='flatten')
        fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
        if is_training:
                fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, 
                            scope='dropout6')
        fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
        if is_training:
                fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, 
                            scope='dropout7')
        grad = tf.gradients(fc6,pool5_flat,name='grad')
        return fc7,grad

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

                rois, roi_scores = _proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")#2000x5
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

                _proposal_targets['rois'] = rois#128,5
                _proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")#128,1
                _proposal_targets['bbox_targets'] = bbox_targets#128,8
                _proposal_targets['bbox_inside_weights'] = bbox_inside_weights#128,8
                _proposal_targets['bbox_outside_weights'] = bbox_outside_weights#128,8

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
        _anchor_targets['rpn_labels'] = rpn_labels#1,1,450,38
        _anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets#1,50,38,36
        _anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights#1,50,38,36
        _anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights#1,50,38,36

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
        _predictions["cls_pred"] = cls_pred#128
        _predictions["cls_prob"] = cls_prob#128,2
        _predictions["bbox_pred"] = bbox_pred#128,8

        return cls_prob, bbox_pred   

def network(is_training=True):
        global anchors
        #with tf.variable_scope('network') as scope:
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

        fc7,grad = vgg16_tail(pool5, is_training)

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

def get_variables_in_checkpoint_file(file_name):
        try:
          reader = pywrap_tensorflow.NewCheckpointReader(file_name)
          var_to_shape_map = reader.get_variable_to_shape_map()
          return var_to_shape_map 
        except Exception as e:  # pylint: disable=broad-except
          print(str(e))
          if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
    
def get_variables_to_restore(variables, var_keep_dic):
        variables_to_restore = []
        _variables_to_fix=[]
    
        for v in variables:
          # exclude the conv weights that are fc weights in vgg16
          if v.name == (_scope + '/fc6/weights:0') or \
             v.name == (_scope + '/fc7/weights:0'):
            # _variables_to_fix[v.name] = v
            continue
          # exclude the first conv layer to swap RGB to BGR
        #   if v.name == (_scope + '/conv1/conv1_1/weights:0'):
            # _variables_to_fix[v.name] = v
            # continue
          if v.name.split(':')[0] in var_keep_dic:
            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)
    
        return variables_to_restore
    


rois, cls_prob, bbox_pred =network(True)


with tf.variable_scope('LOSS') as scope:

        rpn_cls_score = tf.reshape(_predictions['rpn_cls_score_reshape'], [-1, 2])
        rpn_label = tf.reshape(_anchor_targets['rpn_labels'], [-1])
        rpn_select = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])#256,2
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])#256
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # RPN, bbox loss
        rpn_bbox_pred = _predictions['rpn_bbox_pred']
        rpn_bbox_targets = _anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = _anchor_targets['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = _anchor_targets['rpn_bbox_outside_weights']
        rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,rpn_bbox_outside_weights, sigma=3.0, dim=[1, 2, 3])

        # RCNN, class loss
        cls_score = _predictions["cls_score"]#128,2
        label = tf.reshape(_proposal_targets["labels"], [-1])#128
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
 
#         # RCNN, bbox loss
#         bbox_pred = _predictions['bbox_pred']
#         bbox_targets = _proposal_targets['bbox_targets']
#         bbox_inside_weights = _proposal_targets['bbox_inside_weights']
#         bbox_outside_weights = _proposal_targets['bbox_outside_weights']
#         loss_box = _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

#         _losses['cross_entropy'] = cross_entropy
#         _losses['loss_box'] = loss_box
#         _losses['rpn_cross_entropy'] = rpn_cross_entropy
#         _losses['rpn_loss_box'] = rpn_loss_box

#         loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
#         # regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
#         # _losses['total_loss'] = loss + regularization_loss
#         #暂未加l2正则


#         _event_summaries.update(_losses)
       

# decay_rate = 0.5

# decay_steps = 1000
 
# global_step = tf.Variable(0)  

# learning_rate = tf.train.exponential_decay(0.01, global_step, decay_steps, decay_rate, staircase=True)

# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

_scope = 'vgg_16'
variables = tf.global_variables()
print('variabels',variables)
pretrained_model = os.path.join('C:\\Users\\Lenovo\\Desktop\\DL\\Faster_rcnn\\nets','vgg_16.ckpt')
var_keep_dic = get_variables_in_checkpoint_file(pretrained_model)

for key in var_keep_dic:
    print("tensor_name: ", key)

variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
print('111')
print(variables_to_restore)




# with tf.Session() as sess:
#     sess.run(
#         (tf.global_variables_initializer(),
#         tf.local_variables_initializer()
#     ))
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess,coord)

#     training_rounds = 2000000
#     for i in range(training_rounds):
#         image_xs,label_xs,imgsize_xs= sess.run([image_train,box_label_train,image_size_train])
#         sess.run(optimizer,feed_dict={image_placeholder:image_xs,label_placeholder:label_xs,imgsize_placeholder:imgsize_xs})
#         if i%100 ==0:
#             print('i=',i)
#             loss = sess.run(loss,feed_dict={image_placeholder:image_xs,label_placeholder:label_xs,imgsize_placeholder:imgsize_xs})
#             print('loss',loss)
#             print(sess.run(learning_rate))
            
#             image_test_xs, label_test_xs, imgsize_test_xs= sess.run([image_test, box_label_test,image_size_test])
#             test_loss = sess.run(loss,feed_dict={image_placeholder:image_test_xs,label_placeholder:label_test_xs,imgsize_placeholder:imgsize_test_xs})
#             print('test_loss',test_loss)
#     coord.request_stop()
#     coord.join(threads)
#b = tf.get_default_graph().get_tensor_by_name("vgg_16/conv5/conv5_1/weights:0")
with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(),tf.local_variables_initializer()))
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, pretrained_model)
        # sess.run((tf.global_variables_initializer(),tf.local_variables_initializer()))

        out = sess.run(rpn_loss_box ,feed_dict={image_placeholder:a,imgsize_placeholder:b,label_placeholder:c})
        
        print(out)
        print(out.shape)
        #print(sess.run(b))