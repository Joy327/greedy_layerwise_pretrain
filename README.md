# greedy_layerwise_pretrain
cross_model_first.py

        训练第一层encoder-decoder，保存训练的参数，parameters/
cross_model_second.py

        训练第二层encoder-decoder，保存训练的参数，parameters_layer2/
generate_second_data.py

        生成第二层训练的输入数据，也就是在第一层训练完成后，用第一层的参数生成数据作为第二层的输入
model.py

        将两层的训练模型，整合在一起，做encoder-decoder训练。在model中对于两个模型参数的加载，需要两者分开，在声明的新图中，去做参数的恢复

        g1 = tf.Graph()
        with g1.as_default():
             conv_weights_layer1 = tf.get_variable("conv_weights_layer1", [3, 3, 3, 3])
             deconv_weights_layer1 = tf.get_variable("deconv_weights_layer1", [3, 3, 3, 3])
             saver_layer1 = tf.train.Saver({"conv_weights_layer1": conv_weights_layer1,
                                   "deconv_weights_layer1": deconv_weights_layer1})
                with tf.Session(graph=g1) as isess_1:
                saver_layer1.restore(isess_1,'parameters/para.ckpt')
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
                        conv_weights_2_numpy = conv_weights_layer2.eval()
                        deconv_weights_2_numpy = deconv_weights_layer2.eval()



        g3 = tf.Graph()
        model
