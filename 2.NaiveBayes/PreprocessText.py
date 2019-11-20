import jieba
import re
import readFiles


from collections import Counter
class BayesPreprocess():
    def __init__(self,text,stop_words):
        self.text = text
        self.stop_words = stop_words
        self.vocab = set()

    def for_one_list(self):
        content_for_all_file = []  # 将所有文件存放在一个列表中
        for content in self.text:
            content = delet_non_Chinese(content)
            content = word_segmention(content,self.stop_words)
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
def delet_non_Chinese(content):
        # 正则表达式去除非中文
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        content = rule.sub("", content)

        return content
def word_segmention(content,stop_words):
        word_list = []
        content = list(jieba.cut(content))
        for word in content:
            if word not in stop_words:
                word_list.append(word)
        return word_list