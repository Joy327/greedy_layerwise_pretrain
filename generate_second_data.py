import tensorflow as tf
import numpy as np
import cv2
import glob
learning_rate = 0.01
iteration = 600
path = 'dataset/image/00Original/'
img_train_data = []
for filename_path in glob.glob(r'dataset/image/00Original/*.tif'):
    img_train_data.append(cv2.imread(filename_path))
img_train_data = np.array(img_train_data)

'''
get all the variables in the cross_model_test.py model .
and only restore the conv_weights_layer1 
'''
input_x = tf.placeholder(tf.float32)
conv_weights_layer = tf.get_variable("conv_weights_layer",[3,3,3,3],trainable=False)
saver = tf.train.Saver({'conv_weights_layer1':conv_weights_layer})
second_input = tf.nn.conv2d(input_x,conv_weights_layer,strides=[1,1,1,1],padding='SAME')
with tf.Session() as sess:
    saver.restore(sess,'parameters/para.ckpt')
    second = sess.run(second_input,feed_dict={input_x:img_train_data})
    np.save("second.npy",second)
    print("sucessfully generate the second encoder input data")



