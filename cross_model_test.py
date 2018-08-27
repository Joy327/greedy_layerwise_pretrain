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


def normalize(train):
    min,max = train.min(),train.max()
    train = (train-min)/(max-min)
    return train
img_train_data = normalize(img_train_data)
print(img_train_data.dtype)
input_x = tf.placeholder(tf.float32,[None,256,256,3])

conv_weights_layer1 = tf.get_variable("conv_weights_layer1",[3,3,3,3],\
                                      initializer=tf.random_uniform_initializer(minval=-1,maxval=1))
deconv_weights_layer1 = tf.get_variable("deconv_weights_layer1",[3,3,3,3],\
                                    initializer=tf.random_uniform_initializer(minval=-1,maxval=1))


with tf.name_scope('encoder_conv1'):
    encoder = tf.nn.conv2d(input_x,conv_weights_layer1,strides=[1,1,1,1],padding='SAME')

with tf.name_scope('decoder_conv1'):
    output = tf.nn.conv2d(encoder,deconv_weights_layer1,strides=[1,1,1,1],padding='SAME')

with tf.name_scope('train_cost'):
    cost = tf.losses.mean_squared_error(input_x,output)
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(iteration):
        _,c = sess.run([train,cost],feed_dict={input_x : img_train_data})
        if step%10==0:
            print("step is "+str(step)+"------------------cost is : "+str(c))
    print("successfully trained")
    saver = tf.train.Saver()
    save_path = saver.save(sess,"parameters/para.ckpt")
    print("parameters have been saved")
