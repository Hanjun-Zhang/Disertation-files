import tensorflow as tf 
import numpy as np
import tensorflow.contrib.slim as slim
import cv2
from tensorflow.python import pywrap_tensorflow

def vgg16_head(input,is_training):
    with tf.variable_scope('vgg_16') as scope:
        net = slim.repeat(input, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      
        net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        print(net.shape)
        net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool5')
        print(net.shape)
        return net

def face_detection(input,is_training,dropout_keep_prob=0.9):
    with tf.variable_scope('face_detection') as scope:
        net = slim.conv2d(input, 1024, [1, 1] ,padding ='VALID', scope='fc6')
      
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
        net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
        
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, 2, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')

        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net

def bbox_reg(input,is_training,dropout_keep_prob=0.9):
    with tf.variable_scope('bbox_reg') as scope:
        net = slim.conv2d(input, 1024, [1, 1] ,padding ='VALID', scope='fc6')
      
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
        net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
        
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, 4, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')

        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net

def create_net(input,is_training):
        heat_map = vgg16_head(input,is_training)
        out_1 = face_detection(heat_map,is_training)
        # out_2 = bbox_reg(heat_map,is_training)

        return out_1#, out_2

def compute_loss(out_1,out_2,face_label,face_label_2,targets):
        face_prediction = tf.nn.softmax(out_1)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(face_label*tf.log(face_prediction + 1e-10),reduction_indices=[1]))
        reg_loss = tf.reduce_sum(face_label_2*tf.square(tf.subtract(out_2,targets)))

        correct_prediction = tf.equal(tf.argmax(face_prediction,1), tf.argmax(face_label,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = cross_entropy + reg_loss
        return loss,cross_entropy,reg_loss,accuracy

def compute_cross(out_1,face_label):
        face_prediction = tf.nn.softmax(out_1)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(face_label*tf.log(face_prediction + 1e-12),reduction_indices=[1]))
        

        correct_prediction = tf.equal(tf.argmax(face_prediction,1), tf.argmax(face_label,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = cross_entropy
        return loss,accuracy


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
          if v.name == ('vgg_16' + '/fc6/weights:0') or \
             v.name == ('vgg_16' + '/fc7/weights:0'):
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

