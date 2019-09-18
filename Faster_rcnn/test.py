import tensorflow as tf 
import numpy as np

import cv2

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('save_1/model.ckpt.meta')
    saver.restore(sess, 'save_1/model.ckpt')# .data文件
    # pred = tf.get_collection('network-output')[0]
    cls_score = tf.get_collection('cls_score')[0]
    cls_prob = tf.get_collection('cls_prob')[0]
    bbox_pred = tf.get_collection('bbox_pred')[0]
    rois = tf.get_collection('rois')[0]

    graph = tf.get_default_graph()

    image_placeholder = graph.get_operation_by_name('image_placeholder').outputs[0]
    imgsize_placeholder = graph.get_operation_by_name('imgsize_placeholder').outputs[0]
    # direction_placeholder = graph.get_operation_by_name('direction_placeholder').outputs[0]
    # dropout_placeholder = graph.get_operation_by_name('dropout').outputs[0]

    # y = sess.run(pred, feed_dict={image_placeholder: test_x, direction_placeholder: test_y})
#########################################################################################################
    # image_path = 'G:\\HeadPoseImageDatabase\\Person15\\person15168+30-60.jpg'
    # image = cv2.imread(image_path,1)

    # image = cv2.resize(image,(256,256))
    # image = image/255
    # image = image[np.newaxis,:,:,:]
    # print(image.shape)
    image_path = 'G:\\wider_face\\WIDER_test\\WIDER_test\\images\\9--Press_Conference\\9_Press_Conference_Press_Conference_9_889.jpg'
    image = cv2.imread(image_path,1)
    image = cv2.resize(image,(800,600))
    image = image/255
##########################################################################################################
    b = np.array([600,800,3])
    feed_dict = {image_placeholder:image, imgsize_placeholder: b}
    cls_score, cls_prob, bbox_pred, rois = sess.run([cls_score,cls_prob,bbox_pred,rois],feed_dict=feed_dict)
    print(rois.shape)