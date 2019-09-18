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

def pose_estimation(input,is_training,dropout_keep_prob=0.9):
    with tf.variable_scope('bbox_reg') as scope:
        net = slim.conv2d(input, 1024, [1, 1] ,padding ='VALID', scope='fc6')
      
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
        net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
        
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, 3, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')

        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net

def create_net(input,is_training):
        heat_map = vgg16_head(input,is_training)
        out_1 = pose_estimation(heat_map,is_training)
        
        return out_1

def compute_loss(out_1,direction_label):
        sub = tf.subtract(out_1,direction_label)
        sqr = tf.square(sub)
        add = tf.reduce_sum(sqr,axis = 1)
        add = tf.expand_dims(add,1)
        
        loss_func = tf.reduce_mean(add)
        
        return loss_func



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



def lenet(input,is_training,dropout_keep_prob=0.9):
        with tf.variable_scope('lenet') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')

                net = slim.conv2d(net, 256, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')
                
                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,512,scope='fc2')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout2')

                net = slim.fully_connected(net,3,activation_fn=None,scope='fc3')
                
                return net

def network_a(input,is_training,dropout_keep_prob=0.9):#2 c 2 f
        with tf.variable_scope('network_a') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,3,activation_fn=None, scope='fc2')
                return net


def network_b(input,is_training,dropout_keep_prob=0.9):#3 c 2 f
        with tf.variable_scope('network_b') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')
                
                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,3, activation_fn=None, scope='fc2')
                return net


def network_c(input,is_training,dropout_keep_prob=0.9):#3 c 3 f
        with tf.variable_scope('network_c') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')
                
                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,512,scope='fc2')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout2')

                net = slim.fully_connected(net,3,activation_fn=None, scope='fc3')

                return net

def network_d(input,is_training,dropout_keep_prob=0.9):
        with tf.variable_scope('network_c') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                
                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')
                
                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv4')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool4')

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,512,scope='fc2')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout2')

                net = slim.fully_connected(net,3,activation_fn=None, scope='fc3')

                return net


def network_e(input,is_training,dropout_keep_prob=0.9):
        with tf.variable_scope('network_e') as scope:
                net = slim.conv2d(input, 32, [3, 3] ,padding ='SAME', scope='conv1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                a = slim.conv2d(net,32,[3,3],4,padding='SAME',scope='add1')

                net = slim.conv2d(net, 32, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')

                b = slim.conv2d(net,32,[3,3],2,padding='SAME',scope='add2')
                
                net = slim.conv2d(net, 32, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')

                net = tf.concat([a,b,net],-1)

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,512,scope='fc2')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout2')

                net = slim.fully_connected(net,3,activation_fn=None, scope='fc3')

                return net

def network_c_1(input,is_training,dropout_keep_prob=0.9):#3 c 3 f
        with tf.variable_scope('network_c') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')

                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1_1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')
                
                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,512,scope='fc2')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout2')

                net = slim.fully_connected(net,3,activation_fn=None, scope='fc3')

                return net


def network_c_2(input,is_training,dropout_keep_prob=0.9):#3 c 3 f
        with tf.variable_scope('network_c_2') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')

                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1_1')
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1_2')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')
                
                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,512,scope='fc2')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout2')

                net = slim.fully_connected(net,3,activation_fn=None, scope='fc3')

                return net


def network_b_3(input,is_training,dropout_keep_prob=0.9):#3 c 2 f
        with tf.variable_scope('network_b_1') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2_1')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')
                
                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,3, activation_fn=None, scope='fc2')
                return net


def network_b_4(input,is_training,dropout_keep_prob=0.9):#3 c 2 f
        with tf.variable_scope('network_b_1') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2_1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2_2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')
                
                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,3, activation_fn=None, scope='fc2')
                return net


def network_b_5(input,is_training,dropout_keep_prob=0.9):#3 c 2 f
        with tf.variable_scope('network_b_5') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')
                
                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3_1')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,3, activation_fn=None, scope='fc2')
                return net


def network_b_6(input,is_training,dropout_keep_prob=0.9):#3 c 2 f
        with tf.variable_scope('network_b_6') as scope:
                net = slim.conv2d(input, 64, [3, 3] ,padding ='SAME', scope='conv1')
        
                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool2')
                
                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3_1')

                net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', scope='conv3_2')

                net = slim.max_pool2d(net, [2, 2],padding='SAME', scope='pool3')

                net = slim.flatten(net)

                net = slim.fully_connected(net,512,scope='fc1')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout1')

                net = slim.fully_connected(net,3, activation_fn=None, scope='fc2')
                return net
