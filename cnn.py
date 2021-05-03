#coding:utf-8
import os
import pandas as pd
import numpy as np
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #空间不足 ResourceExhaustedError: OOM when allocating这个报错的意思是你的数据太多了，不能一次放进GPU中，虽然你设置了batch_size，但tensorflow默认是一次把所有数据都放进GPU中。

WORD_VECTOR_DIR = r'E:\分布文件夹LZP\新闻文本分类\机器和深度\models\baike.vectors.bin'

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
# 模型构建：https://blog.csdn.net/weixin_43977647/article/details/105346675

from tensorflow.python.keras.utils.vis_utils import plot_model
from DatasetLoader import getdata_souhu_jieba


'''[summary]

[description]
'''
def main():
	# 载入数据集
	word_index, X_train, y_train, X_val, y_val, X_test, y_test = getdata_souhu_jieba(MAX_SEQUENCE_LENGTH)
	
	print('3. #############载入word2vec...#############')

	print('4. #############模型及训练...#############')
	from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
	from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
	from tensorflow.keras.models import Sequential

	model = Sequential()
	# 使用分析出的词个数
	model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
	model.add(Dropout(0.2)) #https://zhuanlan.zhihu.com/p/156825903 为什么使用1d卷积
	# model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1)) #{node sequential/conv1d/conv1d}} = Conv2D[T=DT_FLOAT, data_format="NHWC" 报错解决：https://www.huaweicloud.com/articles/f1e2e9a64ebbb97f21b50582e7a01566.html
	model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1)) #data_format='channels_first'
	model.add(MaxPooling1D(3))
	model.add(Flatten())
	model.add(Dense(EMBEDDING_DIM, activation='relu'))
	model.add(Dense(y_train.shape[1], activation='softmax'))
	model.summary()  #WARNING:tensorflow:Model was constructed with shape (None, 100) for input Tensor("embedding_input:0", shape=(None, 100), dtype=float32), but it was called on an input with incompatible shape (None, 1).
	# from tensorflow.keras.utils import plot_model 报错：ImportError: ('Failed to import pydot解决:https://blog.csdn.net/mr_page/article/details/115914592
	os.environ["PATH"] += os.pathsep + "f:/Graphviz/bin/"
	from tensorflow.python.keras.utils.vis_utils import plot_model
	plot_model(model, to_file='cnn_model.png',show_shapes=True) #保存模型结构图：https://blog.csdn.net/yuwenqi123456/article/details/87633106 添加环境变量-dot测试/不加快捷方式/pip

	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop', #
				  metrics=['acc'])
	print(model.metrics_names)
	model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=128)
	model.save('models/cnn_1ep.h5')

	print('5. #############评估...#############')
	print(model.evaluate(X_test, y_test))

if __name__ == "__main__":



	# 测试Tokenzier
	texts = ["你好 我好 你好 你好 你好 我们 大家 都 好 吗 吗 吗 吗 吗", "分词器 训练 文档 训练 文档 文档 你好 我好"]
	tokenizer = tf.keras.preprocessing.text.Tokenizer(split=" ") #默认也是空格分割的单词
	tokenizer.fit_on_texts(texts)
	fre = tokenizer.word_counts  # 统计词频
	print("type(fre):\n",type(fre))
	print("fre:\n",fre)
	# 查看每个词的词频
	for i in fre.items():
		print(i[0], " : ", i[1])
	# 对词频进行排序
	new_fre = sorted(fre.items(), key = lambda i:i[1], reverse = True)
	print("new_fre:\n",new_fre)
	# # 根据词频进行了升序的排序（注意，词频越大，value越小，这个value不是词频，而是按顺序排列的数字）
	order = tokenizer.word_index
	print("type(order):\n",type(order))
	print("order:\n",order)


	main()

