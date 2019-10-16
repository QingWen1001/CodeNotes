import numpy as np
from sklearn import svm
def load_dataset():
    data_batch, label_batch = [], []
    with open('data.txt')as f:
        for line in f.readlines():
            line = line.split()
            data_batch.append(line[:2])
            label_batch.append(line[2])

    return data_batch,label_batch

class SVM():
    def __init__(self):
        ''''''
    def forward(self):
        ''''''
    def loss(self):
        ''''''
    def updata_weight(self):
        ''''''
    def delta_weight(self):
        ''''''