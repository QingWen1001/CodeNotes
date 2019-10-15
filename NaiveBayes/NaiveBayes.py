from PreprocessText import Preprocess
import readFiles
import numpy as np
import pickle
from collections import Counter
class NaiveBayes():
    def __init__(self,normal_path,spam_path,stop_words_path):
        self.normal_path = normal_path
        self.spam_path = spam_path
        self.stop_word_path = stop_words_path

    def process_train_file(self,path):
        print("读取地址"+path+'中的文件')
        text = readFiles.get_data_from_document(path) # 读取列表内的文件内容

        n_file = len(readFiles.get_filename_list(path)) # 获取文件数量
        print('文件读取完毕，开始处理文件！')

        process = Preprocess(text,self.stop_word_path) # PreprocessText对象实例化
        content = process.for_one_list() # 将所有文件进行预处理，然后合并成一个文件
        process.statistics(content) # 统计单词的词频和概率
        word_freq = process.wrod_fraq # 获取单词的词频 返回类型为 {key：value}
        print(path+"文件处理完毕")
        return n_file,word_freq
    def train(self):
        print("开始模型训练！")
        print(self.normal_path)
        n_normal, self.normal_word_freq = self.process_train_file(self.normal_path)
        n_spam, self.spam_word_freq = self.process_train_file(self.spam_path)
        self.normal_prob = n_normal/(n_normal + n_spam)
        self.spam_prob = n_spam/(n_normal + n_spam)
        print('模型训练完毕！')
    def save(self):
        with open("./model/model_normal_freq.pickle",'wb') as f:
            pickle.dump(self.normal_word_freq,f)

        with open("./model/model_spam_freq.pickle", 'wb') as f:
            pickle.dump(self.spam_word_freq, f)

        with open("./model/model_prob.pickle", 'wb') as f:
            pickle.dump([self.normal_prob,self.spam_prob], f)

    def load(self):
        with open("./model/model_normal_freq.pickle",'rb') as f:
            self.normal_word_freq = pickle.load(f)
        with open("./model/model_spam_freq.pickle",'rb') as f:
            self.spam_word_freq = pickle.load(f)
        with open("./model/model_prob.pickle",'rb') as f:
            data = pickle.load(f)
            self.normal_prob,self.spam_prob = data[0],data[1]
def test_prediction(text, model):
    text = Preprocess.delet_non_Chinese(text)
    text = Preprocess.word_segmention(text)
    bayes_compute(text, model)


def bayes_compute(text,model):
    normal_word_count = counter_word(text, set(model.normal_word_freq))
    spam_word_count = counter_word(text, set(model.spam_word_freq))
    prob_normal = prob_compute(normal_word_count,model.normal_word_freq,model.normal_prob)
    prob_spam = prob_compute(spam_word_count,model.spam_word_freq,model.spam_prob)
    prob_t = np.exp(prob_normal)/(np.exp(prob_normal)+np.exp(prob_spam))
    prob_f = np.exp(prob_spam)/(np.exp(prob_normal)+np.exp(prob_spam))
    return  prob_t

def prob_compute(word_count,model_word_freq,type_prob):
    prob = 0
    i = 0
    for w,c in word_count.items():
        #print(w,c)
        prob += c*np.log(model_word_freq[w])

        i +=1
        if i>=5:
            break
    prob += np.log(type_prob)

    return prob

def counter_word(text,model_set):
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

#def softmax():


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
    preprocess = Preprocess([1,2] , stop_words_path)
    true = 0
    n_test_files = len(test_filename_list)
    for name in test_filename_list:
        # 处理文本
        path = test_path + '/' + name
        content = readFiles.read_file(path)
        content = preprocess.delet_non_Chinese(content)
        content = preprocess.word_segmention(content)
        # 计算概率
        prob = bayes_compute(content,model)
        print(name + ' is normal '+str(prob))
        if int(name)<=1000 and prob > 0.5:
        # 测试样本名字小于1000的是正样本
            true +=1
    print("测试完毕，正确率是：")
    print(true/n_test_files)






    print()