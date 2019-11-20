from PreprocessText import *
import readFiles
import numpy as np
import pickle
from collections import Counter
class NaiveBayes():
    def __init__(self,normal_path,spam_path,stop_words_path):
        self.normal_path = normal_path
        self.spam_path = spam_path
        self.stop_words = set(readFiles.readline(stop_words_path))

    def process_train_file(self,path):
        print("读取地址"+path+'中的文件')
        text = readFiles.get_data_from_document(path) # 读取列表内的文件内容

        n_file = len(readFiles.get_filename_list(path)) # 获取文件数量
        print('文件读取完毕，开始处理文件！')

        self.process = BayesPreprocess(text,self.stop_words) # PreprocessText对象实例化
        content = self.process.for_one_list() # 将所有文件进行预处理，然后合并成一个文件
        self.process.statistics(content) # 统计单词的词频和概率
        word_freq = self.process.wrod_fraq # 获取单词的词频 返回类型为 {key：value}
        n_words = self.process.n_words
        print(path+"文件处理完毕")
        return n_file,word_freq,n_words
    def train(self):
        print("开始模型训练！")
        print(self.normal_path)
        n_normal, self.normal_word_freq ,self.n_normal_words = self.process_train_file(self.normal_path)
        n_spam, self.spam_word_freq, self.n_spam_words = self.process_train_file(self.spam_path)
        self.normal_prob = n_normal/(n_normal + n_spam)
        self.spam_prob = n_spam/(n_normal + n_spam)

        print('模型训练完毕！')
    def save(self):
        with open("./model/model_normal_freq.pickle",'wb') as f:
            pickle.dump(self.normal_word_freq,f)

        with open("./model/model_spam_freq.pickle", 'wb') as f:
            pickle.dump(self.spam_word_freq, f)

        with open("./model/model_prob.pickle", 'wb') as f:
            pickle.dump([self.normal_prob,self.n_normal_words,self.spam_prob,self.n_spam_words], f)
    def load(self):
        with open("./model/model_normal_freq.pickle",'rb') as f:
            self.normal_word_freq = pickle.load(f)
        with open("./model/model_spam_freq.pickle",'rb') as f:
            self.spam_word_freq = pickle.load(f)
        with open("./model/model_prob.pickle",'rb') as f:
            data = pickle.load(f)
            self.normal_prob,self.n_normal_words,self.spam_prob,self.n_spam_words = data[0],data[1],data[2],data[3]

    def test_prediction(self, text):
        text = delet_non_Chinese(text)
        text = word_segmention(text,self.stop_words)

        normal_path = self.bayes_compute(text)
        return normal_path
    def bayes_compute(self,text):

        word_count = self.counter_word(text)

        n_words = self.n_normal_words + self.n_spam_words

        prob_normal = self.prob_compute(word_count, self.normal_word_freq, self.normal_prob, n_words)
        prob_spam = self.prob_compute(word_count, self.spam_word_freq, self.spam_prob, n_words)
        #使用 log 方法时的返回值
        prob = prob_normal - prob_spam
        if prob >0:return 0.6
        else:return 0.4
        # 使用乘法计算时的返回值
        #return prob_normal / (prob_spam + prob_normal)


    def smoothing_add_one(self):
        ''' add-one smoothing'''
    def smoothing_add_k(self):
        ''' add-k smoothing '''
    def smoothing_good_turning(self):
        ''' good-turning smoothing '''
    def prob_compute(self, word_count, model_word_freq, type_prob, tpye_n_words):
        prob = 1
        i = 15
        '''
        for w,c in word_count.items():     
            if w in model_word_freq:
                print(w,' is ',model_word_freq[w])
                prob += c*np.log(model_word_freq[w])
            else:
                # 此处应使用 smoothing 方法 暂用 0.0001代替未出现的词
                prob += c*np.log(1/tpye_n_words)
        '''
        ''' 统计15个词，最多统计10个字典里有的，剩下的字典里没有的由 smoothing 方法补充'''
        for w, c in word_count.items():
            if w in model_word_freq:
                # print(w, ' is ', model_word_freq[w],' prob is '+str(prob))
                prob += c*np.log(model_word_freq[w])
                #prob *= model_word_freq[w] ** c
                i -= 1
            if i <= 5: break
        for w, c in word_count.items():
            if w not in model_word_freq:
                # print(w, ' is ', str(1/tpye_n_words), ' prob is ' + str(prob))
                prob += c*np.log(1/tpye_n_words)
                #prob *= (1 / tpye_n_words) ** c
                i -= 1
            if i == 0: break

        # print('type_prob is ' +str(type_prob))
        #prob *= type_prob
        prob += np.log(type_prob)
        return prob
    def counter_word(self, text):
        ''' 统计单词出现的次数 '''
        word_count = {}
        for word in text:
            if word in word_count:
                word_count[word] += 1
            if word not in word_count:
                word_count[word] = 1
        return word_count


'''
def bayes_and_compute(text,model):
    
    #本方法希望通过计算样本文本中的词与训练数据词典的交集来进行计算。
    #目的是为了避免训练集未出现的新词似的概率为 0
    #1、文本与正文本数据集的交集 计算 正文本的数据
    #2、文本与负文本数据集的交集 计算 负文本的数据集
    #3、交集可能是空集，所以结果做softmax
    #问题：
    #1、交集的大小不一致，没有比较的基础
    #2、交集可能为 0
    
    
    # 统计文本中 与 正\负 文本数据集 交集
    normal_word_count = counter_and_word(text, set(model.normal_word_freq))
    spam_word_count = counter_and_word(text, set(model.spam_word_freq))
    # 计算概率
    prob_normal = prob_compute(normal_word_count,model.normal_word_freq,model.normal_prob)
    prob_spam = prob_compute(spam_word_count,model.spam_word_freq,model.spam_prob)
    #将概率 softmax 方式概率为0情况的出现
    prob_t = np.exp(prob_normal)/(np.exp(prob_normal)+np.exp(prob_spam))
    prob_f = np.exp(prob_spam)/(np.exp(prob_normal)+np.exp(prob_spam))
    return  prob_t
def counter_and_word(text,model_set):
    
    #用来统计交集单词的次数
  
    vocab = set(text)
    same_word_vocab = vocab & model_set
    same_word_count = {}
    for word in text:
        if word in model_set and word in same_word_count:
            #print(same_word_count)
            same_word_count[word] += 1
        if word in model_set and word not in same_word_count:
            same_word_count[word] = 1
    return same_word_count
'''

if __name__== "__main__":
    stop_words_path = './data/中文停用词表.txt'
    normal_path = './data/normal'
    spam_path = './data/spam'
    test_path = './data/test'

    # 模型训练
    model = NaiveBayes(normal_path,spam_path,stop_words_path)
    #model.train()
    #model.save()
    model.load()
    print(model.normal_word_freq)
    print(model.spam_word_freq)

    print('开始测试!')
    test_filename_list = readFiles.get_filename_list(test_path)
    #preprocess = Preprocess([1,2] , stop_words_path)
    true = 0
    n_test_files = len(test_filename_list)
    TP,FP,TN,FN=0,0,0,0
    for name in test_filename_list:
        # 处理文本
        path = test_path + '/' + name
        content = readFiles.read_file(path)
        #content = preprocess.delet_non_Chinese(content)
        #content = preprocess.word_segmention(content)

        # 计算概率
        prob = model.test_prediction(content)
        #prob = bayes_compute(content,model)
        print('The prob of normal for'+name + ' is '+str(prob))
        # 测试样本名字小于1000的是正样本
        if int(name)<1000 :
            if prob > 0.5:
                TP +=1
            else:
                FP +=1
        else:
            if prob > 0.5:
                FN +=1
            else:
                TN +=1

    print("测试完毕， 准确率是：" + str((TP+TN)/(TP+TN+FN+FP)))
    print('精确率'+str(TP/(TP+FP)))
    print('召回率'+str(TP/(FN+TP)))

'''
最后实验结果：

准确率是：0.9076923076923077

正样本
精确率0.844559585492228
召回率0.9644970414201184

负样本
精确率0.9695431472081218
召回率0.8642533936651584

'''
