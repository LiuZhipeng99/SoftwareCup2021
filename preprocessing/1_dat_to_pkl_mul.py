#coding:utf-8
import sys
import pandas as pd
import time
from collections import Counter
# 1.本程序预处理搜狐新闻集并生成数据集：dataframe格式保存为bz2文件，三列：内容，标题，标签
"""
数据格式为

<doc>

<url>页面URL</url>

<docno>页面ID</docno>

<contenttitle>页面标题</contenttitle>

<content>页面内容</content>

</doc>
"""
SOHU_DIR = "E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\news_sohusite_xml.dat"
LABEL_MAPPING = {'auto.sohu.com/': '汽车', 'business.sohu.com/': '财经', 'it.sohu.com/': '科技', 'house.sohu.com': '房产', 'sports.sohu.com/': '体育', 
                     'mil.news.sohu.com/': '军事', 'yule.sohu.com/': '娱乐', 'learning.sohu.com/': '教育', 'game.sohu.com/': '游戏' # map里星号不能通配, '*': "其他" 
                     ,'roll.sohu.com/': '未分类'
                }

def task(doc_xml)->list: #有返回非list的bug未查明强制list了
    if (len(doc_xml.split('<url>')) != 2) or (len(doc_xml.split('<docno>')) != 2) or (len(doc_xml.split('<contenttitle>')) != 2) or (len(doc_xml.split('<content>')) != 2):
        return
    url = doc_xml.split('<url>')[1].split('</url>')[0]
    # docno = doc_xml.split('<docno>')[1].split('</docno>')[0]
    title = doc_xml.split('<contenttitle>')[1].split('</contenttitle>')[0]
    content = doc_xml.split('<content>')[1].split('</content>')[0]
    label = ''
    for key,val in LABEL_MAPPING.items():
        if key in url:
            label = val
            if label=='房产': print(title) #检查下url是不是房产/url失效
            break #没有break就全是其他
    else:
        label = '其他'
    # 人工标记点数据/但有污染的风险(加了前置条件减少了风险)
    if label in ['未分类','其他']:
        if '足' in title :
            label = '体育'
        elif '房' in title:
            label = '房产'
    # doc = {'url':url,'title':title, 'content':content}
    # docs.loc[loci] = [url,title,content]
    # loci += 1 太慢了不知道几个小时能不能跑完，下面list转就一分钟保存四五分钟
    if label!='未分类':  #在这里先清洗了
        return [title,content,label]

def run_imap_mp(func, argument_list, num_processes='', is_tqdm=True):
    '''
    多进程与进度条结合

    param:
    ------
    func:function
        函数
    argument_list:list
        参数列表(每个元素代表一个task参数)
    num_processes:int
        进程数，不填默认为总核心-3
    is_tqdm:bool
        是否展示进度条，默认展示
    ''' 
    result_list_tqdm = []
    try:
        import multiprocessing
        if num_processes == '':
            num_processes = multiprocessing.cpu_count()
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

@exeTime
def data_process():
    # xml_str = open("E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\news_sohusite_xml.dat",'r',encoding='latin-1').read() #latin能解码但是非中文；调用read()会一次性读取文件的全部内容，如果文件有10G，内存就爆了，所以，要保险起见，可以反复调用read(size)方法，每次最多读取size个字节的内容。另外，调用readline()可以每次读取一行内容，调用readlines()一次读取所有内容并按行返回list。
    xml_str = open(SOHU_DIR,'r',encoding='gb18030').read() #结果和ultra转码utf结果一样；其中GB18030兼容GBK和GB2312编码，在处理简体中文的时候，可以统一使用GB18030来读取GBK或者GB2312的文档。
    # print(len(xml)) 1537763850
    print(type(xml_str))
    docs_xml = xml_str.split('<doc>\n')
    del xml_str
    print((docs_xml[2])) #1411997数组
    del docs_xml[0] # 数组第一个是空
    docs = [] #docs从数组换成dataframe
    # loci = 0 #除了用loc也可以用append（series）直接append list不行

    docs = run_imap_mp(task,docs_xml)

    # from multiprocessing import Pool,cpu_count
    # pool = Pool(cpu_count())
    # # docs = pool.map(task,docs_xml) #直接取返回值不能应对有空值的情况
    # for i in docs_xml:
    #     ret = pool.map(task,(i,)) #这种阻塞总的cpu利用率和单进程差不多
    #     # ret = pool.map_async(task,(i,)) 
    #     if ret:
    #         docs.append(ret)
    # #加不加join有什么用
    # print(len(docs))
    # pool.close()
    # pool.join()
    print(len(docs),docs[0])
    def not_empty(s):
        # return s and s.strip()
        return s
    st = time.time()
    docs = (list(filter(not_empty, docs)))
    print(time.time()-st) #0.1秒还行就不用考虑直接map以外的多进程调用了
    # fout = open("news_sohusite.json",'w',encoding="utf8") 字典数组转换为json文件多达4g内存易爆，想办法dataframe存成pkl
    # fout.write(json.dumps(docs))
    # fout.close()
    df = pd.DataFrame(docs, columns=('contenttitle', 'content','label'))
    # df['url'] = df['url'].map(LABEL_MAPPING) # 需要赋值/需要MAPPING有完整的字符串所以只能在for里处理



    print(Counter(df["label"])) #Counter({'其他': 935079, '科技': 199871, '汽车': 138576, '娱乐': 50138, '体育': 44537, '财经': 27489, '教育': 13012, '军事': 3294}) 没有游戏和房产
    # 处理后{'其他': 205350, '科技': 199290, '汽车': 136364, '体育': 73985, '娱乐': 48388, '财经': 25746, '房产': 20736, '教育': 12892, '军事': 3238}/其他类减少于是加上个未分类的条件
    # Counter({'其他': 205350, '科技': 199738, '汽车': 137308, '体育': 71671, '娱乐': 49514, '财经': 27333, '房产': 18815, '教育': 12980, '军事': 3280})/加条件
    # Counter({'其他': 209132, '科技': 199871, '汽车': 138576, '体育': 52411, '娱乐': 50138, '财经': 27489, '房产': 18815, '教育': 13012, '军事': 3294}) /目前只差游戏了
    tesstop
    df.to_pickle("E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\data_process\\news_souhusite.pkl.bz2") 
    # 结果上json4g变成了500m的bz2(pkl文件是2g)

if __name__=="__main__":
    print("1. 预处理搜狐新闻数据集保存为pkl...")
    data_process()

# --------------------4.29/23.
# 游戏部分原始数据往这里加
# 也可以用已分词的加在后面一部分
# ----------------------4.29
# 下了群里哥们的game发现不好分title和主题 暂时不用
# 发现官方提供的示例数据是千多条game虽然有点少还是加进来
    # 直接读pkl而不是读原始数据
    print("2. 添加补充的游戏数据...")
    df = pd.read_pickle("E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\data_process\\news_souhusite.pkl.bz2")
    game = pd.read_csv("E:\分布文件夹LZP\新闻文本分类\机器和深度\data_process\game.csv",header=0,names=['content','label','contenttitle']) #关于header和names参数使用
    print(game)
    df = pd.concat([df,game],ignore_index=True)
    print(Counter(df['label']))
    df.to_pickle("E:\\分布文件夹LZP\\新闻文本分类\\机器和深度\\data_process\\news_souhusite.pkl.bz2") 
