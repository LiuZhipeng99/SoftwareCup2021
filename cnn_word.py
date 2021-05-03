#coding:utf-8
import os
import pandas as pd
import numpy as np
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # ResourceExhaustedError: OOM when allocating这个报错的意思是你的数据太多了，不能一次放进GPU中，虽然你设置了batch_size，但tensorflow默认是一次把所有数据都放进GPU中。
# os.environ["CUDA_VISIBLE_DEVICES"] = 0 # ResourceExhaustedError: OOM when allocating这个报错的意思是你的数据太多了，不能一次放进GPU中，虽然你设置了batch_size，但tensorflow默认是一次把所有数据都放进GPU中。

DATA = "E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\data_process\\news_dataset_jieba.pkl.bz2"
WORD_VECTOR_DIR = r'E:\分布文件夹LZP\新闻文本分类\机器和深度\models\baike.vectors.bin'

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
# 模型构建：https://blog.csdn.net/weixin_43977647/article/details/105346675

from tensorflow.python.keras.utils.vis_utils import plot_model
from DatasetLoader import getdata_souhu_jieba


def train():
	# 载入数据集
	word_index, X_train, y_train, X_val, y_val, X_test, y_test = getdata_souhu_jieba(MAX_SEQUENCE_LENGTH)
	
	print('3. #############载入word2vec...#############')
	import gensim
	from tensorflow.python.keras.utils.vis_utils import plot_model
	w2v_model = gensim.models.KeyedVectors.load_word2vec_format(WORD_VECTOR_DIR, binary=True)
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	not_in_model = 0
	in_model = 0
	for word, i in word_index.items(): 
		if (word) in w2v_model:
			in_model += 1
			embedding_matrix[i] = np.asarray(w2v_model[(word)], dtype='float32')
		else:
			not_in_model += 1
	print(str(not_in_model)+' words not in w2v model')
	from tensorflow.keras.layers import Embedding
	embedding_layer = Embedding(len(word_index) + 1,
								EMBEDDING_DIM,
								weights=[embedding_matrix],
								input_length=MAX_SEQUENCE_LENGTH,
								trainable=False)


	print('4. #############模型及训练...#############')
	from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
	from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
	from tensorflow.keras.models import Sequential

	model = Sequential()
	# 使用上面构造的layer
	model.add(embedding_layer)
	model.add(Dropout(0.2))
	model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1)) 
	model.add(MaxPooling1D(3))
	model.add(Flatten())
	model.add(Dense(EMBEDDING_DIM, activation='relu'))
	model.add(Dense(y_train.shape[1], activation='softmax'))
	model.summary()  
	os.environ["PATH"] += os.pathsep + "f:/Graphviz/bin/"
	from tensorflow.python.keras.utils.vis_utils import plot_model
	plot_model(model, to_file='cnn_model.png',show_shapes=True)

	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop', #
				  metrics=['acc'])
	print(model.metrics_names)
	model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=128)
	model.save('./models/cnn_word.h5')

	print('4. #############评估...#############')
	print(model.evaluate(X_test, y_test))

		
if __name__ == "__main__":
	train()
    # model = tf.keras.models.load_model('models/cnn_word.h5')

	

