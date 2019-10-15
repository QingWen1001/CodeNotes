import jieba
import re
import readFiles
import pickle

#from NaiveBayes import readFiles
from collections import Counter
class Preprocess():
    def __init__(self,text,stop_words_path):
        self.text = text
        self.stop_words = set(readFiles.readline(stop_words_path))
        self.vocab = set()
    def delet_non_Chinese(self,content):
        # 正则表达式去除非中文
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        content = rule.sub("", content)

        return content
    def word_segmention(self,content):
        word_list = []
        content = list(jieba.cut(content))
        for word in content:
            if word not in self.stop_words:
                word_list.append(word)
        return word_list
    def for_one_list(self):
        content_for_all_file = []  # 将所有文件存放在一个列表中
        for content in self.text:
            content = self.delet_non_Chinese(content)
            content = self.word_segmention(content)
            content_for_all_file.extend(content)
            words = Counter(content_for_all_file)
            content_for_all_file = [word for word in content_for_all_file if words[word] > 2]  # 去除低频词

        return content_for_all_file
    def word2id(self,text):
        self.vocab = set(text)
        self.vocab2id = {w:c for c,w in enumerate(list(self.vocab))}
        self.id2vocab = {c:w for c,w in enumerate(list(self.vocab))}
    def statistics(self,text):
        self.word_count = Counter(text)
        self.n_words = len(text)
        self.wrod_fraq = {w:c/self.n_words for w,c in self.word_count.items()}

    def save(self):
        with open('model.pickle','wb') as f:
            #pickle.dump(self.wrod_fraq,f)
            pickle.dump([123123,13123],f)
    def load(self):
        with open('model.pickle','rb') as f:
            text = pickle.load(f)
            return text
if __name__ == "__main__":
    text = readFiles.read_file('./data/normal/201')
    stop_words_path = './data/中文停用词表.txt'
    pross = Preprocess(text,stop_words_path)

    text = pross.delet_non_Chinese(text)
    text = pross.word_segmention(text)

    #pross.statistics(text)

    #pross.save()
    #a = pross.load()
    #print(pross.wrod_fraq)
    #print(a)
    #print(a[1])
    #print(pross.wrod_fraq)

    #print(text)
    #print(pross.stop_words)
