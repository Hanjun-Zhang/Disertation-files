
import tensorflow as tf 
import os 
import scipy.misc
import numpy as np
import random
import cv2
from utils.cython_bbox import bbox_overlaps
from model.bbox_transform import bbox_transform


# np.set_printoptions(threshold=np.inf)
root = 'G:\\HeadPoseImageDatabase'
writer = tf.python_io.TFRecordWriter(os.path.join('G:\\HeadPoseImageDatabase\\list\\head_10.tfrecords'))
image_list_doc = os.path.join(root,'list','list.txt')
file_1 = open(image_list_doc)
lines = len(file_1.readlines())#185184
print(lines)


with open (image_list_doc,'r') as f:
        imgs = f.read().splitlines()

for k in range(1):
    image_path,x1,y1,x2,y2 = imgs[k].split(' ')
    # print(image_path,x1,y1,x2,y2)
    label_sub = np.array([int(x1),int(y1),int(x2),int(y2)])
    label_sub = np.reshape(label_sub,[1,4])
    # print(label_sub.shape)
    label = label_sub
    for m in range(230):
        label = np.vstack((label,label_sub))
    # print('111111')
    # print(label.shape)
    
    image = cv2.imread(image_path,1)
    image = cv2.resize(image,(400,300))
    print(image.shape)
    # image = image[100:300,100:300]
    # cv2.imshow('1',image)
    # cv2.waitKey(0)
    image_batch = []
    image_area = []
    for i in range(21):
        for j in range(11):
            sub_image = image[10*j:10*j+200,10*i:10*i+200,:]#[y:y+h,x:x+w]
            sub_image = cv2.resize(sub_image,(32,32))
            sub_image = sub_image[np.newaxis,:,:,:]
            area = np.array([10*i,10*j,10*i+200,10*j+200])#[x,y]
            area = area[np.newaxis,:]
            # print(area)
            # print(sub_image.shape)
            image_batch.append(sub_image)
            image_area.append(area)
    image_batch = np.concatenate(image_batch,axis=0)
    anchors = np.concatenate(image_area,axis=0)
    # print(anchors.shape)
    print(image_batch.shape)
    print(image_batch.dtype)

    targets = bbox_transform(anchors, label)
    # print('11111111111')
    # print(targets,targets.shape,targets.dtype)
    print(targets.shape)
    # print(targets)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(label_sub, dtype=np.float))
    # print(overlaps)
    
    n = np.where(overlaps[:,0] > 0.7)
    print(n)
    face_label_1 = np.ones([231,1])
    face_label_2 = np.zeros([231,1])
    face_label_2[n,0]=1
    face_label_1[n,0]=0
    face_label = np.concatenate([face_label_1,face_label_2],axis=1)
    # print('face_label')
    print(face_label)
    print(face_label_2.shape)
    # print(face_label.dtype)

    img_raw = image_batch.tostring()
    face_label_raw = face_label.tostring()
    face_label_2_raw = face_label_2.tostring()
    targets_raw = targets.tostring()

    example = tf.train.Example(
        features = tf.train.Features (
            feature = {
                'image_batch':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'face_label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[face_label_raw])),
                'face_label_2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[face_label_2_raw])),
                'targets':tf.train.Feature(bytes_list=tf.train.BytesList(value=[targets_raw]))
            }
        )
    )
                        
    writer.write(example.SerializeToString())



