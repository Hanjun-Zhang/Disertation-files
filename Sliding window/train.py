import tensorflow as tf
import numpy as np
from network import vgg16_head,face_detection,bbox_reg, create_net,compute_loss,compute_cross, get_variables_in_checkpoint_file,get_variables_to_restore
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import cv2
import tensorflow.contrib.slim as slim
import os

from tensorflow.python import pywrap_tensorflow


train_path = '\xxxx.tfrecords'
test_path = '\xxxx.tfrecords'


def read_and_decode(file_path):
    files = tf.train.match_filenames_once(file_path)
    filename_queue = tf.train.string_input_producer(files,num_epochs=None,shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_batch':tf.FixedLenFeature([],tf.string),
            'face_label':tf.FixedLenFeature([],tf.string),
            'face_label_2':tf.FixedLenFeature([],tf.string),
            'targets':tf.FixedLenFeature([],tf.string)
        }
    )
    image = tf.decode_raw(features['image_batch'],tf.uint8)
    image = tf.reshape(image,[216,32,32,3])
    image = tf.cast(image,tf.float32)/255
    # image = tf.image.per_image_standardization(image)

    face_label = tf.decode_raw(features['face_label'],tf.float64)
    face_label = tf.reshape(face_label,[216,2])
    face_label = tf.cast(face_label,tf.float32)

    face_label_2 = tf.decode_raw(features['face_label_2'],tf.float64)
    face_label_2 = tf.reshape(face_label_2,[216,1])
    face_label_2 = tf.cast(face_label_2,tf.float32)

    targets = tf.decode_raw(features['targets'],tf.float64)
    targets = tf.reshape(targets,[216,4])
    targets = tf.cast(targets,tf.float32)
    return image,face_label,face_label_2,targets

image_train,face_label_train,face_label_2_train,targets_train = read_and_decode(train_path)

# image_test,box_label_test = read_and_decode(test_path)


image_placeholder = tf.placeholder(tf.float32,[None,32,32,3],name='image_placeholder')
face_placeholder = tf.placeholder(tf.float32,[None,2],name='face_placeholder')
face_2_placeholder = tf.placeholder(tf.float32,[None,1],name='face_2_placeholder')
targets_placeholder = tf.placeholder(tf.float32,[None,4],name='targets_placeholder')

out_1= create_net(image_placeholder,True)

# loss,cross_entropy,reg_loss,accuracy = compute_loss(out_1,out_2,face_placeholder,face_2_placeholder,targets_placeholder)
loss,accuracy = compute_cross(out_1,face_placeholder)

decay_rate = 0.5
decay_steps = 100 
global_step = tf.Variable(0)  
learning_rate = tf.train.exponential_decay(0.0001, global_step, decay_steps, decay_rate, staircase=True)
# optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cross_entropy,global_step=global_step)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)


saver = tf.train.Saver()

_scope = 'vgg_16'
variables = tf.global_variables()
print('variabels',variables)
pretrained_model = os.path.join('C:\\Users\\Lenovo\\Desktop\\DL\\Headpose\\eval\\nets','vgg_16.ckpt')
var_keep_dic = get_variables_in_checkpoint_file(pretrained_model)

for key in var_keep_dic:
    print("tensor_name: ", key)

variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
print('111')
print(variables_to_restore)

with tf.Session() as sess:

    sess.run(
            (tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ))
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, pretrained_model)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)

    training_rounds = 5580
    for i in range(training_rounds):
        image_xs,face_label_xs,face_label_2_xs,targets_xs = sess.run([image_train,face_label_train,face_label_2_train,targets_train])
        sess.run(optimizer,feed_dict={image_placeholder:image_xs,face_placeholder:face_label_xs,face_2_placeholder:face_label_2_xs,targets_placeholder:targets_xs})
        # print(image_xs.shape)
        # print(face_label_xs.shape)
        # print(face_label_2_xs.shape)
        # print(targets_xs.shape)
        if i%10 ==0:
            print('i=',i)
            a,b= sess.run([loss,accuracy],feed_dict={image_placeholder:image_xs,face_placeholder:face_label_xs,face_2_placeholder:face_label_2_xs,targets_placeholder:targets_xs})
        # print(output.shape)
            print(a,b)
    saver.save(sess,'save_3/model.ckpt')
    coord.request_stop()
    coord.join(threads)
