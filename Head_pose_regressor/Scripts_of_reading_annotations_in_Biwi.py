import tensorflow as tf 
import os 
import cv2
import numpy as np
import random
import struct
import math

image_list_doc = 'G:\\kinect_head_pose\\kinect_head_pose_db\\hpdb\\01\\frame_00481_pose.txt'

with open (image_list_doc,'r') as f:
        lines = f.read().splitlines()

ro = np.zeros([3,3])

for i in range(3):
        r1,r2,r3,_ = lines[i].split(' ')
        ro[i,0] = r1
        ro[i,1] = r2
        ro[i,2] = r3
print(ro)


x = math.atan2(ro[2,1], ro[2,2])# pitch下为正
print(x)
a = math.sqrt(math.pow(ro[2,1],2) + math.pow(ro[2,2],2))
y = math.atan2(ro[2,0] ,a)# yaw右为正
print(y)
z = math.atan2(ro[1,0] ,ro[0,0])#raw 
print(z)

image_path = 'G:\\kinect_head_pose\\kinect_head_pose_db\\hpdb\\02\\frame_00300_rgb.png'
image = cv2.imread(image_path,1)
print(image,image.shape)
image = image[150:350,250:450,:]
cv2.imshow('1',image)
cv2.waitKey(0)


txtfile = open(os.path.join("G:\\kinect_head_pose\\kinect_head_pose_db",'list_24.txt'),'w')

for filename in os.listdir(os.path.join('G:\\kinect_head_pose\\kinect_head_pose_db\\hpdb','24')):
        if filename[-3:] =='txt':
                print(filename)
                image_name = filename.replace('pose.txt','rgb.png')
                print(image_name)
                image_path = os.path.join('G:\\kinect_head_pose\\kinect_head_pose_db\\hpdb','24',image_name)
                print(image_path)
                                
                image_list_doc = os.path.join('G:\\kinect_head_pose\\kinect_head_pose_db\\hpdb','24',filename)

                with open (image_list_doc,'r') as f:
                        lines = f.read().splitlines()

                ro = np.zeros([3,3])

                for i in range(3):
                        r1,r2,r3,_ = lines[i].split(' ')
                        ro[i,0] = r1
                        ro[i,1] = r2
                        ro[i,2] = r3
                print(ro)

                x = math.atan2(ro[2,1], ro[2,2])# pitch下为正
                print(x)
                a = math.sqrt(math.pow(ro[2,1],2) + math.pow(ro[2,2],2))
                y = math.atan2(ro[2,0] ,a)# yaw右为正
                print(y)
                z = math.atan2(ro[1,0] ,ro[0,0])#roll
                print(z)
                
                str_x = str(x)
                str_y = str(y)
                str_z = str(z)
                print(str_x,str_y,str_z)
                txtfile.write(image_path)
                txtfile.write(' ')
                txtfile.write(str_x)
                txtfile.write(' ')
                txtfile.write(str_y)
                txtfile.write(' ')
                txtfile.write(str_z)
                txtfile.write('\n')
