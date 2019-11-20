# Code Notes
项目中复现了一些经典的机器学习算法，并通过简单的任务展示了算法或模型的效果。在一些算法或模型的复现的过程中，一些模型的对比总结，公式推导，数学原理也存在了项目文件夹中。
主要工具：pycharm，jupyter notebook，pytorch
以下是项目的目录。项目持续更新中...
### 1. LogisticRegression（LR）
- LR 逻辑回归的复现
- Logistic Regression Summery 总结了逻辑回归、线性回归的数学原理公式推导，比较了两种方法的异同和特点，介绍了决策边界、代价函数、优化方法和正则项的引入。
- Image of LR ：不同参数，不同优化方法的分类结果
### 2. NaiveBayes
- data 数据集
- model 保存训练好的模型参数
- readFiles 读取数据
- PreprocessText 数据的预处理
- NaivesBayes 使用朴素贝叶斯模型进行垃圾邮件分类
- 朴素贝叶斯 VS 逻辑回归 ：对比总结了朴素贝叶斯方法与逻辑回归特点和异同
- 生成模型 VS 判别模型 ：对比总结了生成模型与判别模型的特点和异同，公式推导，解释为了什么有生成模型可以得到判别模型而从判别模型得不到生成模型。
### 3. HMM
- NER by HMM ：使用HMM模型进行命名实体识别的序列标注任务
- HMM在实现过程中的一些问题的总结：在模型复现的过程中 对于未出现字的处理，HMM 模型的缺点。

### 4. SupportVectorMachine(SVM)
### 5. CRF
- sklearn 工具的 CRF 模型的使用
## PyTorch
pytorch学习笔记，使用PyTorch复现一些经典的深度学习算法。
### 1.双层神经网络
- 介绍了 tensor 的使用方法，numpy 与 tensor 之间的转换
- CUDA 调用 GPU 进行计算的方式
- 使用 numpy 构建简单的神经网络并训练
- 通过一步一步的改进，将 numpy 构建的神经网络改进成使用 pytorch 构建的神经网络
- 学习如何使用 pytorch 自定义一个神经网络模型。
### 2.word-embedding
- 2.0 word-embedding 使用 pytorch 复现了使用 skip-gram 方法进行 word2vec 计算的模型。
### 3.LSTM
- 3.0 LSTM model ：使用pytorch 复现了 LSTM 模型
- 3.1 RNN-LSTM-GRU-model ：使用pytorch复现了 LSTM RNN GRU 模型
### 4.Sentiment
- 使用 Word Averaging 模型，LSTM 模型进行情感计算，主要流程是先获得句子的向量表示，在使用 linear 层进行分类。
- Text CNN 模型： 通过句子的词向量进行排列，组成类似图像的矩阵，然后使用 filter 进行卷积计算，可以很好的将不同的长度的句子处理成相同长度。
