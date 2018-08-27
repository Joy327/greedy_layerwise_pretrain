import tensorflow as tf
import numpy as np
learning_rate = 0.01
iteration = 600
data_set = np.load('second.npy')

def normalize(train):
    min,max = train.min(),train.max()
    train = (train-min)/(max-min)
    return train
data_set = normalize(data_set)

input_second = tf.placeholder(tf.float32,[None,256,256,3])

with tf.name_scope("encoder_conv2"):
    conv_weights_layer2 = tf.get_variable('conv_weights_layer2',[3,3,3,3],\
                                      initializer=tf.random_uniform_initializer(minval=-1,maxval=1))
with tf.name_scope("decoder_conv2"):
    deconv_weights_layer2 = tf.get_variable('deconv_weights_layer2',[3,3,3,3],\
                                    initializer=tf.random_uniform_initializer(minval=-1,maxval=1))
second_encoder_input = tf.nn.max_pool(input_second,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
print(second_encoder_input.shape)
encoder = tf.nn.conv2d(second_encoder_input,conv_weights_layer2,strides=[1,1,1,1],padding='SAME')
print(encoder.shape)
decoder = tf.nn.conv2d(encoder,deconv_weights_layer2,strides=[1,1,1,1],padding='SAME')
print(decoder.shape)
cost_2 = tf.losses.mean_squared_error(second_encoder_input,decoder)
train_2 = tf.train.AdamOptimizer(learning_rate).minimize(cost_2)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(iteration):
        _,c_2 = sess.run([train_2,cost_2],feed_dict={input_second:data_set})
        if step%10 == 0:
            print("step is "+str(step)+"-------------cost is : "+str(c_2))
    print("successfully trained")
    saver = tf.train.Saver()
    save_path = saver.save(sess,"parameters_layer2/para_layer2.ckpt")
    print("parameters have been saved")


