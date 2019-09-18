import tensorflow as tf 
import numpy as np
from network_only_reg import vgg16_head,compute_loss,lenet,network_a,network_b,network_c,network_d,network_e,network_c_1,network_c_2,network_b_3
from network_only_reg import network_b_4,network_b_5,network_b_6
# from model.bbox_transform import bbox_transform
# from utils.cython_bbox import bbox_overlaps
import cv2
import tensorflow.contrib.slim as slim
import os

from tensorflow.python import pywrap_tensorflow


train_path = 'G:\\kinect_head_pose\\kinect_train_32.tfrecords'
val_path = 'G:\\kinect_head_pose\\kinect_val_32.tfrecords'
logs_path="./net_b_6"

def read_and_decode(file_path):
    files = tf.train.match_filenames_once(file_path)
    filename_queue = tf.train.string_input_producer(files,num_epochs=None,shuffle=True)

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

image_train,direction_train = read_and_decode(train_path)
image_val,direction_val = read_and_decode(val_path)

min_after_dequeue = 300
batch_size = 128
capacity = min_after_dequeue + 3 * batch_size

image_train_batch, direction_train_batch = tf.train.shuffle_batch(
    [image_train,direction_train],batch_size = batch_size, capacity = capacity, min_after_dequeue = min_after_dequeue,num_threads=2
)


image_val_batch, direction_val_batch = tf.train.shuffle_batch(
    [image_val,direction_val],batch_size = batch_size, capacity = capacity, min_after_dequeue = min_after_dequeue,num_threads=2
)


image_placeholder = tf.placeholder(tf.float32,[None,32,32,3],name='image_placeholder')
direction_placeholder = tf.placeholder(tf.float32,[None,3],name='direction_placeholder')

out_1= network_b_6(image_placeholder,True)

# loss,cross_entropy,reg_loss,accuracy = compute_loss(out_1,out_2,face_placeholder,face_2_placeholder,targets_placeholder)
loss = compute_loss(out_1,direction_placeholder)

decay_rate = 0.5
decay_steps = 560 
global_step = tf.Variable(0)  
learning_rate = tf.train.exponential_decay(0.001, global_step, decay_steps, decay_rate, staircase=True)
# optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cross_entropy,global_step=global_step)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

# tf.summary.scalar("loss", loss)
# merged_summary_op = tf.summary.merge_all()

training_summary = tf.summary.scalar("training_loss", loss)
validation_summary = tf.summary.scalar("validation_loss", loss)

saver = tf.train.Saver()



with tf.Session() as sess:

    sess.run(
            (tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ))
    
    summary_writer=tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)

    training_rounds = 10000
    for i in range(training_rounds):
        image_xs,direction_xs = sess.run([image_train_batch,direction_train_batch])
        sess.run(optimizer,feed_dict={image_placeholder:image_xs,direction_placeholder:direction_xs})
        # print(image_xs.shape)
        # print(face_label_xs.shape)
        # print(face_label_2_xs.shape)
        # print(targets_xs.shape)
        if i%100 ==0:
            print('i=',i)
            a,b= sess.run([loss,training_summary],feed_dict={image_placeholder:image_xs,direction_placeholder:direction_xs})
            # summary=sess.run(merged_summary_op,feed_dict={image_placeholder:image_xs,direction_placeholder:direction_xs})
            summary_writer.add_summary(b, i)
        # print(output.shape)
            
            image_val_xs,direction_val_xs = sess.run([image_val_batch,direction_val_batch])
            c,d= sess.run([loss,validation_summary],feed_dict={image_placeholder:image_val_xs,direction_placeholder:direction_val_xs})
            # summary=sess.run(merged_summary_op,feed_dict={image_placeholder:image_xs,direction_placeholder:direction_xs})
            summary_writer.add_summary(d, i)
            print(a,c)
    saver.save(sess,'net_b_6/model.ckpt')
    coord.request_stop()
    coord.join(threads)
summary_writer.close()

