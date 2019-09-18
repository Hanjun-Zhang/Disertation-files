import tensorflow as tf 
import numpy as np



a = np.random.random((16,224,224,3))
b = np.random.random((16,3))
c = np.random.random((16,2))

labels = np.empty((3,), dtype=np.float32)
labels.fill(-1)
print(labels.shape)

overlaps = np.array([[0.5,0.7,0.3,0.4,0.1],[0.6,0.7,0.8,0.9,0.5]]).T
print(overlaps.shape)
argmax_overlaps = overlaps.argmax(axis=1)
print('argmax_overlaps',argmax_overlaps,argmax_overlaps.shape)
max_overlaps = overlaps[np.arange(5), argmax_overlaps]
print('max_overlaps',max_overlaps)
gt_argmax_overlaps = overlaps.argmax(axis=0)
gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]
print('gt_max_overlaps',overlaps)
gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
print('gt_argmax_overlaps',gt_argmax_overlaps)

# labels[max_overlaps < 0.3] = 0
# labels[gt_argmax_overlaps] = 1
# labels[max_overlaps >= 0.7] = 1

gt_boxes = np.array([100,110,200,220,1])
print(gt_boxes[argmax_overlaps, :])