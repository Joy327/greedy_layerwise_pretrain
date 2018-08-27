import tensorflow as tf
import numpy as np
import glob
import cv2
conv_weights_1_numpy = np.zeros(())
deconv_weights_1_numpy = np.zeros(())
conv_weights_2_numpy = np.zeros(())
deconv_weights_2_numpy = np.zeros(())

g1 = tf.Graph()
with g1.as_default():
    conv_weights_layer1 = tf.get_variable("conv_weights_layer1", [3, 3, 3, 3])
    deconv_weights_layer1 = tf.get_variable("deconv_weights_layer1", [3, 3, 3, 3])
    saver_layer1 = tf.train.Saver({"conv_weights_layer1": conv_weights_layer1,
                                   "deconv_weights_layer1": deconv_weights_layer1})
    with tf.Session(graph=g1) as isess_1:
        saver_layer1.restore(isess_1,'parameters/para.ckpt')
        print(conv_weights_layer1)
        print(deconv_weights_layer1)
        conv_weights_1_numpy = conv_weights_layer1.eval()
        deconv_weights_1_numpy = deconv_weights_layer1.eval()
g2 = tf.Graph()
with g2.as_default():
    conv_weights_layer2 = tf.get_variable("conv_weights_layer2", [3, 3, 3, 3])
    deconv_weights_layer2 = tf.get_variable("deconv_weights_layer2", [3, 3, 3, 3])
    saver_layer2 = tf.train.Saver({"conv_weights_layer2": conv_weights_layer2,
                                   "deconv_weights_layer2": deconv_weights_layer2})
    with tf.Session(graph=g2) as isess_2:
        saver_layer1.restore(isess_2, 'parameters_layer2/para_layer2.ckpt')
        print(conv_weights_layer2)
        print(deconv_weights_layer2)
        conv_weights_2_numpy = conv_weights_layer2.eval()
        deconv_weights_2_numpy = deconv_weights_layer2.eval()



g3 = tf.Graph()
with g3.as_default():
    learning_rate = 0.01
    iteration = 600
    path = 'dataset/image/00Original/'
    img_train_data = []
    for filename_path in glob.glob(r'dataset/image/00Original/*.tif'):
        img_train_data.append(cv2.imread(filename_path))
    img_train_data = np.array(img_train_data)
    input_x = tf.placeholder(tf.float32, [None, 256, 256, 3])
    def normalize(train):
        min, max = train.min(), train.max()
        train = (train - min) / (max - min)
        return train
    img_train_data = normalize(img_train_data)
    print("training data is sucessfully loaded and normalize.  the img_train_data shape is : " + str(
        img_train_data.shape))
    def UnPooling2x2ZeroFilled(x):
        # https://github.com/tensorflow/tensorflow/issues/2169
        out = tf.concat([x, tf.zeros_like(x)], 3)
        out = tf.concat([out, tf.zeros_like(out)], 2)
        sh = x.get_shape().as_list()
        if None not in sh[1:]:
            out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
            return tf.reshape(out, out_size)
        else:
            shv = tf.shape(x)
            ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
            return ret
    conv_weights_1 = tf.get_variable("conv_weights_1", [3, 3, 3, 3],trainable=True)
    deconv_weights_1 = tf.get_variable("deconv_weights_1", [3, 3, 3, 3],trainable=True)
    conv_weights_2 = tf.get_variable("conv_weights_2", [3, 3, 3, 3],trainable=True)
    deconv_weights_2 = tf.get_variable("deconv_weights_2", [3, 3, 3, 3],trainable=True)
    encoder_layer1 = tf.nn.conv2d(input_x,conv_weights_1,strides=[1,1,1,1],\
                              padding='SAME')
    maxpooling_layer1 = tf.nn.max_pool(encoder_layer1,ksize=[1,2,2,1],strides=[1,2,2,1],\
                                   padding='VALID')
    encoder_layer2 = tf.nn.conv2d(maxpooling_layer1,conv_weights_2,strides=[1,1,1,1],\
                              padding='SAME')
    maxpooling_layer2 = tf.nn.max_pool(encoder_layer2,ksize=[1,2,2,1],strides=[1,2,2,1],\
                                   padding='VALID')
    unmaxpooling_layer2 = UnPooling2x2ZeroFilled(maxpooling_layer2)
    decoder_layer2 = tf.nn.conv2d(unmaxpooling_layer2,deconv_weights_2,strides=[1,1,1,1],\
                              padding='SAME')
    unmaxpooling_layer1 = UnPooling2x2ZeroFilled(decoder_layer2)
    decoder_layer1 = tf.nn.conv2d(unmaxpooling_layer1,deconv_weights_1,strides=[1,1,1,1],\
                              padding='SAME')
    cost = tf.losses.mean_squared_error(input_x,decoder_layer1)
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    with tf.Session(graph=g3) as isess_3:
        isess_3.run(tf.global_variables_initializer())
        conv_weights_1 = tf.convert_to_tensor(conv_weights_1_numpy)
        deconv_weights_1 = tf.convert_to_tensor(deconv_weights_1_numpy)
        conv_weights_2 = tf.convert_to_tensor(conv_weights_2_numpy)
        deconv_weights_2 = tf.convert_to_tensor(deconv_weights_2_numpy)
        for step in range(iteration):
            _,c = isess_3.run([train,cost],feed_dict={input_x:img_train_data})
            if step%10 ==0:
                print("after setp  "+str(step)+" ----------- the cost is : "+str(c))
        print("sucessfully trained")
        saver = tf.train.Saver()
        save_path = saver.save(isess_3,"total_parameters/para_total.ckpt")
        print("parameters have been saved")
