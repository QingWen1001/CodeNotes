# coding=utf-8 ##以utf-8编码储存中文字符
from codecs import open
import numpy as np
import pickle
def load_data(path):
    '''将所有文本读取成一个列表'''
    with open(path,'r',encoding='utf-8')as f:
        text = f.read()
        text = text.replace('\n\r', '\n$ $\n') # 人工标记句子开头
        text = text.split()
        data_list =[text[i] for i in range(0,len(text),2)]
        label_list = [text[i] for i in range(1,len(text),2)]
    return data_list, label_list
'''
def load_data_for_list():
    word_lists = []
    tag_lists = []

    with open("test_data.char.bmes", 'r', encoding='utf-8')as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
    return  word_lists,tag_lists
'''
def word_2_id(text):
    ''' 将内容数字化，返回原内容 数字化的list 和 字典'''
    vocab2id = {}
    for i in text:
        if i not in vocab2id:
            vocab2id[i] = len(vocab2id)
    text = [vocab2id[i] for i in text] # 将内容数字化，
    return text, vocab2id

class HMM():
    def __init__(self,word2id,label2id):
        self.word2id = word2id
        self.label2id = label2id
        self.n_vocab = len(self.word2id)
        self.n_state = len(self.label2id)
        self.A = np.zeros((self.n_state,self.n_state)) # 状态转移矩阵
        self.B = np.zeros((self.n_state,self.n_vocab)) # 发射矩阵
        self.PI = np.zeros(self.n_state) # 初始状态
    def train(self,data_list,state_list):
        '''

        :param data_list: word list
        :param state_list: label list
        :param start_id_in_state: the start target id in label2id
        :return:
        '''
        self.train_A(state_list)
        self.train_B(data_list,state_list)
        self.train_PI()
    def train_A(self,state_list):
        ''''''
        for i in range(len(state_list)-1):
            self.A[state_list[i]][state_list[i+1]] +=1

        #将次数转换为概率
        for i in range(self.n_state):
            a = 0
            for j in range(self.n_state):
                if self.A[i][j] == 0 :
                    self.A[i][j] = 1e-10
                a += self.A[i][j]

            if a!=0:
                self.A[i] /= a
        #print(self.A)
        #self.A = self.A / self.A.sum(dim=1, keepdim=True) #将次数转换为概率
    def train_B(self,data_list,state_list):
        ''''''
        for i in range(len(state_list)):
            self.B[state_list[i]][data_list[i]] +=1

        for i in range(self.n_state):
            a = 0
            for j in range(self.n_vocab):
                if self.B[i][j] == 0 :
                    self.B[i][j] = 1e-10
                a += self.B[i][j]
            if a!=0:
                self.B[i] /= a
        #print(self.B)
        #self.B = self.B / self.B.sum(dim=1, keepdim=True)# 将次数转换为概率
    def train_PI(self):
        '''状态转移矩阵中 $ 的转移概率 就是 初始状态的概率'''
        self.PI = self.A[self.label2id['$']]
        #print(self.PI)
    def save(self,path):
        ''' save model '''
        with open(path,'wb')as f:
            data = [self.A,self.B,self.PI,self.word2id,self.label2id,self.n_state,self.n_vocab]
            pickle.dump(data,f)
    def load(self,path):
        ''' load model '''
        with open(path,'rb') as f:
            data = pickle.load(f)
            self.A, self.B, self.PI, self.word2id, self.label2id, self.n_state, self.n_vocab = data
    def viterbi_decoding(self,input_text):
        '''
        使用 viterbi 算法求解 HMM 的状态转移序列
        返回 最佳序列
        '''
        ##  正向计算
        # 初始化 dp 矩阵，存储路径矩阵
        dp = np.zeros((len(input_text),self.n_state))
        best_path = np.zeros((len(input_text),self.n_state))
        # 计算第一个位置
        for i in range(self.n_state):
            dp[0][i] = self.PI[i]*self.get_state2word(input_text[0])[i]
            best_path[0][i] = -1
        # 计算后续位置
        for i in range(1,len(input_text)):
            word = input_text[i]
            for j in range(self.n_state):
                #dp[i][j] = max(dp[i - 1][k] * self.A[k][j] * self.get_state2word(word) for k in range(self.n_state))
                best_state = 0
                for k in range(self.n_state):
                    # 第 i 个字，第 j 个状态的概率 是 第 i-1 个字 到第 i 个字的 状态转移概率 × 状态发射字 的概率。
                    prob = dp[i-1][k]*self.A[k][j]*self.get_state2word(word)[j]
                    if prob > dp[i][j]:
                        dp[i][j] = prob
                        best_state = k
                best_path[i][j] = best_state
        ## 反向回溯
        path = []
        id = np.argmax(dp[len(input_text) - 1])
        path.append(id)
        for i in range(0,len(input_text)):
            id = int(id)
            id = best_path[int(len(input_text)-i-1)][id]
            path.append(id)
        #print(dp)
        #print(best_path)
        path = [path[len(path)-2-i] for i in range(len(path)-1)]
        return path

    def get_state2word(self,word):
        if word in self.word2id:
            id = self.word2id[word]
            return self.B[:,id]
        else:
            return np.ones(self.n_state,1)
if __name__=='__main__':
    data_path ="train_data.txt"
    data_list,label_list = load_data(data_path)
    data_list,data2id = word_2_id(data_list)
    label_list,label2id = word_2_id(label_list)
    id2label = {c:w for w,c in label2id.items()}

    hmm=HMM(word2id=data2id,label2id=label2id)
    hmm.train(data_list=data_list,state_list=label_list)
    hmm.save('./model/model.pickle')
    #hmm.load('./model/model.pickle')

    test = list('常建良是工科学士和总经理')
    target = hmm.viterbi_decoding(test)
    target = [id2label[i] for i in target]

    print({test[i]:target[i] for i in range(len(test))})
