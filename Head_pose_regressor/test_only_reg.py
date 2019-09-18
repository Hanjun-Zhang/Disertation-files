import tensorflow as tf 
import numpy as np
from network_only_reg import vgg16_head,compute_loss,lenet,network_a,network_b,network_c,network_d,network_e
from network_only_reg import network_c_1,network_c_2,network_b_3,network_b_4,network_b_5,network_b_6
# from model.bbox_transform import bbox_transform
# from utils.cython_bbox import bbox_overlaps
import cv2
import tensorflow.contrib.slim as slim
import os

from tensorflow.python import pywrap_tensorflow

# np.set_printoptions(threshold=np.inf)


# image = cv2.imread('G:\\kinect_head_pose\\kinect_head_pose_db\\cut\\21\\frame_00050_rgb.png',1)


# image = cv2.resize(image,(32,32))
# print('111111111111')
# mean = np.mean(image)
# std = np.std(image)
# image = (image-mean)/std
# print(image)
# image = image[np.newaxis,:,:,:]
# image = tf.image.per_image_standardization(image)


# test_path = 'G:\\kinect_head_pose\\kinect_train_32.tfrecords'
test_path = 'G:\\kinect_head_pose\\kinect_test_32.tfrecords'
test_result_a = open('G:\\kinect_head_pose\\test_result_a.txt','w')

def read_and_decode(file_path):
    files = tf.train.match_filenames_once(file_path)
    filename_queue = tf.train.string_input_producer(files,num_epochs=1,shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'images':tf.FixedLenFeature([],tf.string),
            'direction':tf.FixedLenFeature([],tf.string)
        }
    )
    image = tf.decode_raw(features['images'],tf.uint8)
    image = tf.reshape(image,[32,32,3])
    image = tf.cast(image,tf.float32)
    image = tf.image.per_image_standardization(image)

    direction = tf.decode_raw(features['direction'],tf.float64)
    direction = tf.reshape(direction,[3])
    direction = tf.cast(direction,tf.float32)
    return image,direction

image_test,direction_test = read_and_decode(test_path)
image_test = tf.expand_dims(image_test,0)
direction_test = tf.expand_dims(direction_test,0)

# sess = tf.Session()

image_placeholder = tf.placeholder(tf.float32,[None,32,32,3],name='image_placeholder')
direction_placeholder = tf.placeholder(tf.float32,[None,3],name='direction_placeholder')

out_1= network_b_6(image_placeholder,False)
sub = out_1 - direction_placeholder
sess = tf.Session()

sess.run(
        (tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ))
saver = tf.train.Saver()
saver.restore(sess,'net_b_6/model.ckpt')

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess,coord)

i = 0
for i in range(1344):
    image,direction = sess.run([image_test,direction_test])
    head_pose = sess.run(sub,feed_dict={image_placeholder:image,direction_placeholder:direction})
    print(head_pose[0,1])
    i = i+1
    print(i)
    pitch = int(abs(head_pose[0,0])/3.14*180)
    yaw = int(abs(head_pose[0,1])/3.14*180)
    roll = int(abs(head_pose[0,2])/3.14*180)
    # test_result_a.write(str(pitch))
    # test_result_a.write(' ')
    # test_result_a.write(str(yaw))
    # test_result_a.write(' ')
    test_result_a.write(str(roll))
    test_result_a.write('\n')
coord.request_stop()
coord.join(threads)
test_result_a.close()


