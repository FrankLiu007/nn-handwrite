#!/usr/bin/env python

import os
import sys


import numpy as np
##不同环境请，修改home_dir,
# 请尽量设置为绝对路径，保证任何地方跑该代码都不出错
home_dir = r"C:\Users\liuqimin\nn-handwrite"

bin_dir = os.path.join(home_dir, 'bin')
model_dir = os.path.join(home_dir, 'model')
conf_dir = os.path.join(home_dir, 'conf')
data_dir = os.path.join(home_dir, 'data')
lib_dir = os.path.join(home_dir, 'lib')
log_dir = os.path.join(home_dir, 'log')
sys.path.append(lib_dir)

import tokenFile

# 输出单元激活函数
def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))

class myRNN:
    def __init__(self, data_dim, hidden_dim=100, bptt_back=4):
        # data_dim: 词向量维度，即词典长度; hidden_dim: 隐单元维度; bptt_back: 反向传播回传时间长度
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.bptt_back = bptt_back

        # 初始化权重向量 U， W， V; U为输入权重; W为递归权重; V为输出权重
        self.U = np.random.uniform(-np.sqrt(1.0/self.data_dim), np.sqrt(1.0/self.data_dim),
                                   (self.hidden_dim, self.data_dim))
        self.W = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                   (self.hidden_dim, self.hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                   (self.data_dim, self.hidden_dim))

    # 前向传播
    def forward(self, x):
        # 向量时间长度
        T = len(x)

        # 初始化状态向量, s包含额外的初始状态 s[-1]
        s = np.zeros((T+1, self.hidden_dim))
        o = np.zeros((T, self.data_dim))

        for t in range(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))

        return [o, s]

    # 预测输出
    def predict(self, x):
        o, s = self.forward(x)
        pre_y = np.argmax(o, axis=1)
        return pre_y

    # 计算损失， softmax损失函数， (x,y)为多个样本
    def loss(self, x, y):
        cost = 0
        for i in range(len(y)):
            o, s = self.forward(x[i])
            # 取出 y[i] 中每一时刻对应的预测值
            pre_yi = o[range(len(y[i])), y[i]]
            cost -= np.sum(np.log(pre_yi))

        # 统计所有y中词的个数, 计算平均损失
        N = np.sum([len(yi) for yi in y])
        ave_loss = cost / N

        return ave_loss

    # 求梯度, (x,y)为一个样本
    def bptt(self, x, y):
        dU = np.zeros(self.U.shape)
        dW = np.zeros(self.W.shape)
        dV = np.zeros(self.V.shape)

        o, s = self.forward(x)
        delta_o = o
        delta_o[range(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:
            # 梯度沿输出层向输入层的传播
            dV += delta_o[t].reshape(-1, 1) * s[t].reshape(1, -1)  # self.data_dim * self.hidden_dim
            delta_t = delta_o[t].reshape(1, -1).dot(self.V) * ((1 - s[t-1]**2).reshape(1, -1)) # 1 * self.hidden_dim

            # 梯度沿时间t的传播
            for bpt_t in np.arange(np.max([0, t-self.bptt_back]), t+1)[::-1]:
                dW += delta_t.T.dot(s[bpt_t-1].reshape(1, -1))
                dU[:, x[bpt_t]] = dU[:, x[bpt_t]] + delta_t

                delta_t = delta_t.dot(self.W.T) * (1 - s[bpt_t-1]**2)

        return [dU, dW, dV]

    # 计算梯度
    def sgd_step(self, x, y, learning_rate):
        dU, dW, dV = self.bptt(x, y)

        self.U -= learning_rate * dU
        self.W -= learning_rate * dW
        self.V -= learning_rate * dV

    # 训练RNN
    def train(self, X_train, y_train, learning_rate=0.005, n_epoch=5):
        losses = []
        num_examples = 0

        for epoch in range(n_epoch):
            for i in range(len(y_train)):
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples += 1

            loss = self.loss(X_train, y_train)
            losses.append(loss)
            print ('epoch {0}: loss = {1}'.format(epoch+1, loss) )
            # 若损失增加，降低学习率
            if len(losses) > 1 and losses[-1] > losses[-2]:
                learning_rate *= 0.5
                print( 'decrease learning_rate to', learning_rate )


unknown_token = 'UNKNOWN_TOKEN'
start_token = 'START_TOKEN'
end_token = 'END_TOKEN'

def generate_text(rnn, dict_words, index_of_words):
    # dict_words: type list; index_of_words: type dict
    sent = [index_of_words[start_token]]
    # 预测新词，知道句子的结束(END_TOKEN)
    while not sent[-1] == index_of_words[end_token]:
        next_probs, _ = rnn.forward(sent)
        sample_word = index_of_words[unknown_token]

        # 按预测输出分布进行采样，得到新的词
        while sample_word == index_of_words[unknown_token]:
            samples = np.random.multinomial(1, next_probs[-1])
            sample_word = np.argmax(samples)
        # 将新生成的有含义的词(即不为UNKNOWN_TOKEN的词)加入句子
        sent.append(sample_word)

    new_sent = [dict_words[i] for i in sent[1:-1]]
    new_sent_str = ' '.join(new_sent)

    return new_sent_str

if __name__ == '__main__':
    file_path = os.path.join(data_dir, r'reddit-comments-2015-08.csv')
    dict_size = 8000
    myTokenFile = tokenFile.tokenFile2vector(file_path, dict_size)
    X_train, y_train, dict_words, index_of_words = myTokenFile.get_vector()
    rnn = myRNN(dict_size, hidden_dim=100, bptt_back=4)
    rnn.train(X_train[:200], y_train[:200], learning_rate=0.005, n_epoch=10)

    sent_str = generate_text(rnn, dict_words, index_of_words)
    print( 'Generate sentence:', sent_str )