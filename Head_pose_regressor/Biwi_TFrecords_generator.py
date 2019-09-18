import tensorflow as tf 
import os 
import cv2
import numpy as np
import random
import struct
import math


writer = tf.python_io.TFRecordWriter(os.path.join('G:\\kinect_head_pose\\kinect_val_32.tfrecords'))
for filename in os.listdir(os.path.join('G:\\kinect_head_pose\\kinect_head_pose_db\\cut')):
        if filename[-5:] =='l.txt':
                print(filename)
                file_path = os.path.join('G:\\kinect_head_pose\\kinect_head_pose_db\\cut',filename)
                file_1 = open(file_path)
                num = len(file_1.readlines())
                with open (file_path,'r') as f:
                        lines = f.read().splitlines()
                for i in range(num):

                        image_path,pitch,yaw,roll = lines[i].split(' ')
                        print(image_path,pitch,yaw,roll)
                        image_path = image_path.replace('hpdb','cut')
                        img = cv2.imread(image_path,1)
                        img = cv2.resize(img,(32,32))
                        print(img,img.shape)
                        pitch = float(pitch)
                        yaw = float(yaw)
                        roll = float(roll)
                        print(pitch)
                        direction = np.array([pitch,yaw,roll])
                        print(direction,direction.shape,direction.dtype)

                        img_raw = img.tostring()
                        direction_raw = direction.tostring()
                        

                        example = tf.train.Example(
                                features = tf.train.Features (
                                        feature = {
                                                'images':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                                                'direction':tf.train.Feature(bytes_list=tf.train.BytesList(value=[direction_raw]))
                                
                                        }
                                )
                        )
                                
                        writer.write(example.SerializeToString())
