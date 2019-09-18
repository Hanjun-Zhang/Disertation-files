import tensorflow as tf 
import numpy as np 
from nets import vgg
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
import os 

a = np.random.random((1,800,600,3))
image = tf.placeholder(tf.float32,[None,800,600,3])

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
    
        
            return net

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

_scope = 'vgg_16'

out = vgg16_head(image,True)

# pretrained_model = '/nets/vgg_16.ckpt'
variables = tf.global_variables()
print('variabels',variables)
pretrained_model = os.path.join('C:\\Users\\Lenovo\\Desktop\\DL\\transform_learning\\nets','vgg_16.ckpt')
var_keep_dic = get_variables_in_checkpoint_file(pretrained_model)

# for v in variables:
#     variables_to_restore.append(v)
#     print(v.name)
# print(variables_to_restore)

for key in var_keep_dic:
    print("tensor_name: ", key)

variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
print('111')
print(variables_to_restore)

b = tf.get_default_graph().get_tensor_by_name("vgg_16/conv5/conv5_1/weights:0")

sess = tf.Session()
restorer = tf.train.Saver(variables_to_restore)
restorer.restore(sess, pretrained_model)

# result = sess.run(out,feed_dict={image:a})
# print(result,result.shape)
print(sess.run(b))

