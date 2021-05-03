import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# import tensorflow as tf
DATA = "E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\data_process\\news_dataset_jieba.pkl.bz2"

def getdata_souhu_jieba(MAX_SEQUENCE_LENGTH):
	'''第一版数据集读取函数，读取经jieba分词的搜狐新闻源数据 格式如下：
	
	contenttitle|content|label|contenttitle_cut
	为什么人要吃饭|今天 小编 带 大家|0|为什么 人 要 吃饭
	
	Arguments:
		MAX_SEQUENCE_LENGTH {[type]} -- [description]
	
	Returns:
		[type] -- [description]
	'''
	# TODO:模型输出是1*10的词向量；label是数字直接用来分类也要改模型
	print('1. #############载入数据集...#############')
	data = pd.read_pickle(DATA)# 暂时没用到title

	# X_train, X_test, y_train, y_test = train_test_split(df.drop(['Label'],axis=1), df['Label'], test_size=.20, random_state=42)
	# 划分训练集、测试集、验证集
	# X, X_val,y , y_val = train_test_split(data['content'], data['label'], test_size=.20, random_state=1024)
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=1024)
	# print(X_test)
	
	all_label = data['label'] #不知道为什么必须不能直接在下面用

	print('2. #############词向量化/划分数据集...#############')

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(data['content'])
	joblib.dump(tokenizer, 'models/Tokenizer')

	sequences = tokenizer.texts_to_sequences(data['content'])
	word_index = tokenizer.word_index
	print('数据集含 %s 个不同的词.' % len(word_index))
	data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	labels = to_categorical(np.asarray(all_label))
	print('Shape of data tensor:', data.shape,"[0]:", data[0])
	print('Shape of label tensor:', labels.shape,"[0]:", labels[0])


	# 第一部分划分应该放这儿/模型fit报错就是因为直接用的原始数据
	X, X_test, y, y_test = train_test_split(data, labels, test_size=.20, random_state=1024)
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.10, random_state=1024)
	return word_index,X_train,y_train,X_val,y_val,X_test,y_test

def word_to_vector(text, MAX_SEQUENCE_LENGTH):
	tokenizer = joblib.load("./models/Tokenizer")
	word_index = tokenizer.word_index
	sequences = tokenizer.texts_to_sequences(text)
	data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	print('Shape of data tensor:', data.shape,"[0]:", data[0])
	return data

def softmax(L):
	expL = np.exp(L)
	sumExpL = sum(expL)
	result = []
	for i in expL:
		result.append(i*1.0/sumExpL)
	return result
if __name__ == '__main__':
	#测试下预测功能
	import tensorflow as tf
	model = tf.keras.models.load_model('./models/cnn_word')
	# model = tf.saved_model.load('./models/cnn_word') 这个model没有predit之类很麻烦
	# 输入是一段内容
	data = '今日房价暴跌，620寝室贬值'
	import jieba.posseg as pseg
	gen = pseg.cut(data)
	words = []
	for i in gen:
		if i.flag != 'x':
			words.append(i.word)  
	data = ' '.join(words)  
	data = word_to_vector(['北京邮电大学 专业 录取 分数线 排名 高校 学科 具体 专业 科别 平均分 最高分 就业 '],100)
	print(data.shape)
	# data = (data.reshape(1,18,100)) #预测需要加个bathsize
	print(max((model.predict(data)[0]))) #直接model调用
	print((model(data))) #直接model调用区别仅在是不是tensor类型
