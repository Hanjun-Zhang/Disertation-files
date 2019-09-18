import tensorflow as tf 
import os 
import cv2
import numpy as np
import random


root = 'G:\\HeadPoseImageDatabase'
writer = tf.python_io.TFRecordWriter(os.path.join('G:\\HeadPoseImageDatabase\\2418_reg_cv.tfrecords'))
image_list_doc = os.path.join(root,'list.txt')
file_1 = open(image_list_doc)
lines = len(file_1.readlines())

with open (image_list_doc,'r') as f:
        imgs = f.read().splitlines()

for i in range(lines):
        person,angle = imgs[i].split(' ')
        angle = angle.rstrip('.jpg')
        # print(person)
        # print(angle)

        # if angle[12] =='0':
        if angle[-2] =='+':
                if angle[-4] == '+':
                        print('111')
                        tilt = 0
                        pan = 0
                else:
                        pan = 0
                        tilt = angle[-5:-2]
        else:
                if angle[-5] =='+':
                        tilt = 0
                        pan = angle[-3:]
                else:
                        tilt = angle[11:14]
        
                        pan = angle[14:]
        
        
        int_tilt = int(tilt)
        int_pan = int(pan)
        print(person,int_tilt,int_pan)
        
        y = 1*np.sin(int_tilt*np.pi/180)
        x = np.cos(int_tilt*np.pi/180) * np.sin(int_pan*np.pi/180)
        z = np.cos(int_tilt*np.pi/180) * np.cos(int_pan*np.pi/180)

        direction = np.array([x,y,z])

        image_path = os.path.join(root,person,angle+'.jpg')
        
        img = cv2.imread(image_path,1)

        img = cv2.resize(img,(32,32))
        print(img.shape)
        print(direction)
        print(direction.shape)
        print(direction.dtype)
        print(i)
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





