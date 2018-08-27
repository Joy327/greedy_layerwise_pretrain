# greedy_layerwise_pretrain
#cross_model_first.py

        ##训练第一层encoder-decoder，保存训练的参数，parameters/
#cross_model_second.py

        ##训练第二层encoder-decoder，保存训练的参数，parameters_layer2/
#generate_second_data.py

        ##生成第二层训练的输入数据，也就是在第一层训练完成后，用第一层的参数生成数据作为第二层的输入
#model.py

        ##将两层的训练模型，整合在一起，做encoder-decoder训练
