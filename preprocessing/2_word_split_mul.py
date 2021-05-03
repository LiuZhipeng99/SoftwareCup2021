#coding:utf-8
import sys

import jieba.posseg as pseg
import pandas as pd

# import numpy as np
"""
2.本程序进行分词：把内容列和title列分词。并把标签编码（上一部分没做）。输出存入文件，（还是单独写个模块读这个文件）
"""


# ------设置文件变量
news_pkl = "E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\data_process\\news_souhusite.pkl.bz2" #714037条数据,磁盘读取速度10m以下读入需要近一分钟
output_pkl = "E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\data_process\\news_dataset.pkl.bz2"
#5.1：分词title列并保留原来的备用（content列字太多了不用）
#5.1：加入头条数据集；
toutiao_txt = "E:\分布文件夹LZP\新闻文本分类\数据集\toutiao_cat_data.txt"
'''
数据来源：今日头条客户端
数据格式：
6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
每行为一条数据，以`_!_`分割的个字段，从前往后分别是 新闻ID，分类code（见下文），分类名称（见下文），新闻字符串（仅含标题），新闻关键词
分类code与名称：
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu 
109 科技 科技 news_tech
110 军事 军事 news_military
114 证券 股票 stock
116 电竞 游戏 news_game
数据规模：
共382688条
'''
#5.1：
# ------问题
# TODO 还是先取前m个字符再切词还是先切词再取前n个词（权衡性能和速度）：具体做法是更改cut函数在line切片或是在words切片
# TODO 试试北大开源的PKUseg https://cloud.tencent.com/developer/news/383935 观察速度  并试试使用新闻语料库：https://www.jianshu.com/p/7ad0cd33005e
# TODO jieba使用停用词库，细节上简书
# TODO 使用多进程提速/完善进度条的显示
from multiprocessing import Pool,cpu_count
print("可用的CPU数量：",cpu_count())
# TODO 没有清洗搜狐数据集（content为空但默认空字符串），在补充的游戏集有脏数据（content为空且为NaN）。（遵从最小改动原则只改了cut函数将这种情况返回空字符串），空数据清洗应该在1.里面并考虑除去还是如何处理
# ------
# ------方法调研
# https://www.zhihu.com/question/67922726 需不需要中文分词？分标题？内容？还是全部
# https://zhuanlan.zhihu.com/p/47761862 中文分类方法
# https://blog.csdn.net/zhonglongshen/article/details/78845173 比较了词向量字向量onehot
# 分词是需要的（词向量化比字向量化更有意义），如何向量化见后续词嵌入embedding




# map调用的函数：将新闻内容用jieba切词
def content_to_word(line):
    gen = pseg.cut(line)
    words = []
    for i in gen:
        if i.flag != 'x':
            words.append(i.word) #只加非标点符号
    return ' '.join(words) #空格分割的字符串
def content100_to_word(line):
    try:
        gen = pseg.cut(line[:100]) #在最后一万条有报错，line是nan
    except:
        return ""
    words = []
    for i in gen:
        if i.flag != 'x':
            words.append(i.word) 
    return ' '.join(words)
def isString(obj):
    try:
        obj.lower() + obj.title() + obj + ""
    except:
        return False
    else:
        return True

def exeTime( func ):
    '''函数运行时间修饰器
    
    Arguments:
        func {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    '''
    def wrapper( *args, **args2 ):
        t0 = time.time()
        back = func( *args, **args2 )
        t1 = time.time()
        print ('该函数耗时：', t1 - t0 )
        return back
    return wrapper
def run_imap_mp(func, argument_list, num_processes='', is_tqdm=True):
    '''
    多进程与进度条结合
    param:
    ------
    func:function
        函数
    argument_list:list
        参数列表
    num_processes:int
        进程数，不填默认为总核心-3
    is_tqdm:bool
        是否展示进度条，默认展示
    ''' 
    result_list_tqdm = []
    try:
        import multiprocessing
        if num_processes == '':
            num_processes = multiprocessing.cpu_count()-3
        pool = multiprocessing.Pool(processes=num_processes)
        if is_tqdm:
            from tqdm import tqdm
            for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
                result_list_tqdm.append(result)
        else:
            for result in pool.imap(func=func, iterable=argument_list):
                result_list_tqdm.append(result)
        pool.close()
    except:
        result_list_tqdm = list(map(func,argument_list))   
    return result_list_tqdm
    
@exeTime
def main():
    print("1. loading data...")
    df = pd.read_pickle(news_pkl)
    print(df[700000:].all())
    print("2. jieba分词...（此过程在i5七代上半个多小时）")
    # df = df[:100]map是返回拷贝
    df['content'][0:2000] = df['content'][0:2000].map(content100_to_word)
    print("######1%######")
    df['content'][2000:200000] = df['content'][2000:200000].map(content100_to_word) #挺费时,二十万条需要近十分钟
    print("######30%######")
    df['content'][200000:500000] = df['content'][200000:500000].map(content100_to_word)
    print("######60%######")
    df['content'][500000:] = df['content'][500000:].map(content100_to_word)
    print("######90%######")
    df['contenttitle_cut'] = df['contenttitle'].map(content_to_word)
    print(df)

    print("3. Labeling...")
    # from sklearn.preprocessing import LabelEncoder
    # df['label'] = LabelEncoder().fit_transform(df['label']) #返回拷贝
    LABEL_MAPPING = {'其他': '0', '财经': '1', '科技': '2', '房产': '3', '体育': '4', 
                     '汽车': '5', '娱乐': '6', '教育': '7', '游戏': '8' ,"军事":'9'
                } #方便将预测的数字替换
    df['label'] = df['label'].map(LABEL_MAPPING)


    print("4. Save to "+output_pkl+"...")
    # df.to_csv("test.csv",encoding='gbk')
    df.to_pickle(output_pkl)


if __name__=="__main__":
    main()
    # df = pd.read_pickle("E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\data_process\\news_dataset_jieba.pkl.bz2")
    # print(df)
    # LABEL_MAPPING = {'其他': '0', '财经': '1', '科技': '2', '房产': '3', '体育': '4', 
    #                  '汽车': '5', '娱乐': '6', '教育': '7', '游戏': '8' ,"军事": '9'
    #             }
    # df['contenttitle_cut'] = df['contenttitle'].map(content_to_word)
    
    # df['label'] = df['label'].map(LABEL_MAPPING)
    # print(df)
    # df.to_pickle("E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\data_process\\news_dataset_jieba.pkl.bz2")



    # 测试
    # lits = ['test','测试']
    # print('|'.join(lits))
    # import jieba
    # print(list(jieba.cut("不知道这个工具怎么样。我喜欢太阳")))
    # words = ((pseg.cut("不知道这个工具怎么样。我喜欢太阳"))) #words是个generator类需要迭代
    # for w in words:
    #     print(w) #打印出来是’知道/v‘的形式可以单独访问
    #     print(w.word,w.flag)
