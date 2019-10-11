import pandas as pd # process FormData
import matplotlib.pyplot as plt
import numpy as np

# load data
def load_data_from_csv(path):
    '''

    :param path: string
    :return: data_form
    '''
    df = pd.read_csv(path, header=None)
    #print(df.loc[ :, [0,1]])
    return df

def load_data_from_txt(path):
    '''

    :param path: string
    :return: []
    '''
    with open(path) as f:
        data = f.readlines()
    return data

def get_batch(data):
    '''

    :param data: []
    :return: [] []
    '''
    batch_data, batch_label = [], []
    for line in data:
        line = line.strip().split()
        batch_data.append([1.0, float(line[0]), float(line[1])])
        batch_label.append(float(line[2]))
    return batch_data, batch_label

def sigmoid(x):
    return 1.0/(1+np.exp(-x))
class LogisticRegression():
    def __init__(self, data_list, label_list, learn_rate):
        self.learn_rate = learn_rate
        # change the type of data form a list to a matrix
        self.data = np.mat(data_list)
        self.label = np.mat(label_list)

        self.data_dim,self.weight_dim = self.data.shape # Get the dimension of data and weight

        self.weight = np.random.randn(self.weight_dim,1) # The matrix of weight
    def update_weight(self,delt_weight):
        self.weight = self.weight - self.learn_rate*delt_weight

    def GradientDecent(self):
        y = sigmoid(self.data*self.weight) # 100*3 * 3*1 = 100*1
        loss = y - self.label.transpose()  # 100*1 - 100*1 = 100*1
        delt_weight = self.data.transpose()*loss # 3*100 * 100*1 = 3*1
        self.update_weight(delt_weight)

    def StochasitcGradientDencent(self,):
        for i in range(self.data_dim):
            y = sigmoid(self.data[i]*self.weight)
            loss =y - self.label[0,i]
            delt_weight = loss*self.data[i]
            self.update_weight(delt_weight.transpose())

    def plotFinalGraph(self):
        x_core1,x_core2,y_core1,y_core2 = [],[],[],[]
        for i in range(self.data_dim):
            if self.label[:,i] == 1:
                x_core1.append(self.data[i,1])
                y_core1.append(self.data[i,2])
            else:
                x_core2.append(self.data[i,1])
                y_core2.append(self.data[i,2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_core1,y_core1, s=30, c='red', marker='s')
        ax.scatter(x_core2, y_core2, s=30, c="green")
        x = np.arange(-5,5,0.1)
        y = (-self.weight[0,0]- self.weight[1,0]*x)/self.weight[2,0]
        ax.plot(x,y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

    def train(self, Optimization="GradientDecent"):
        if Optimization == "GradientDecent":
            self.GradientDecent()
        elif Optimization == "StochasticGradientDecent":
            self.StochasitcGradientDencent()

if __name__ == "__main__":
    learning_rate = 0.01
    epoch = 1000
    optimization = "StochasticGradientDecent"
    data = load_data_from_txt("data.txt")
    batch_data, batch_label = get_batch(data)

    model = LogisticRegression(batch_data, batch_label,learning_rate)
    for i in range(epoch):
        model.train(Optimization=optimization)
    model.plotFinalGraph()