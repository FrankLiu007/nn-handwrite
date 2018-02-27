#!/usr/bin/env python
# -*- coding: gb18030 -*-
'''
Copyright (c) Baidu.com, Inc. All Rights Reserved

@Time     : 2018/2/27 15:50

@Author   : yangfengguang

@File     : rnn.py

@Software : PyCharm
'''
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np

home_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
bin_dir = os.path.join(home_dir, 'bin')
model_dir = os.path.join(home_dir, 'model')
conf_dir = os.path.join(home_dir, 'conf')
data_dir = os.path.join(home_dir, 'data')
lib_dir = os.path.join(home_dir, 'lib')
log_dir = os.path.join(home_dir, 'log')
sys.path.append(lib_dir)

import tokenFile
# �����Ԫ�����
def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))

class myRNN:
    def __init__(self, data_dim, hidden_dim=100, bptt_back=4):
        # data_dim: ������ά�ȣ����ʵ䳤��; hidden_dim: ����Ԫά��; bptt_back: ���򴫲��ش�ʱ�䳤��
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.bptt_back = bptt_back

        # ��ʼ��Ȩ������ U�� W�� V; UΪ����Ȩ��; WΪ�ݹ�Ȩ��; VΪ���Ȩ��
        self.U = np.random.uniform(-np.sqrt(1.0/self.data_dim), np.sqrt(1.0/self.data_dim),
                                   (self.hidden_dim, self.data_dim))
        self.W = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                   (self.hidden_dim, self.hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                   (self.data_dim, self.hidden_dim))

    # ǰ�򴫲�
    def forward(self, x):
        # ����ʱ�䳤��
        T = len(x)

        # ��ʼ��״̬����, s��������ĳ�ʼ״̬ s[-1]
        s = np.zeros((T+1, self.hidden_dim))
        o = np.zeros((T, self.data_dim))

        for t in xrange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))

        return [o, s]

    # Ԥ�����
    def predict(self, x):
        o, s = self.forward(x)
        pre_y = np.argmax(o, axis=1)
        return pre_y

    # ������ʧ�� softmax��ʧ������ (x,y)Ϊ�������
    def loss(self, x, y):
        cost = 0
        for i in xrange(len(y)):
            o, s = self.forward(x[i])
            # ȡ�� y[i] ��ÿһʱ�̶�Ӧ��Ԥ��ֵ
            pre_yi = o[xrange(len(y[i])), y[i]]
            cost -= np.sum(np.log(pre_yi))

        # ͳ������y�дʵĸ���, ����ƽ����ʧ
        N = np.sum([len(yi) for yi in y])
        ave_loss = cost / N

        return ave_loss

    # ���ݶ�, (x,y)Ϊһ������
    def bptt(self, x, y):
        dU = np.zeros(self.U.shape)
        dW = np.zeros(self.W.shape)
        dV = np.zeros(self.V.shape)

        o, s = self.forward(x)
        delta_o = o
        delta_o[xrange(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:
            # �ݶ���������������Ĵ���
            dV += delta_o[t].reshape(-1, 1) * s[t].reshape(1, -1)  # self.data_dim * self.hidden_dim
            delta_t = delta_o[t].reshape(1, -1).dot(self.V) * ((1 - s[t-1]**2).reshape(1, -1)) # 1 * self.hidden_dim

            # �ݶ���ʱ��t�Ĵ���
            for bpt_t in np.arange(np.max([0, t-self.bptt_back]), t+1)[::-1]:
                dW += delta_t.T.dot(s[bpt_t-1].reshape(1, -1))
                dU[:, x[bpt_t]] = dU[:, x[bpt_t]] + delta_t

                delta_t = delta_t.dot(self.W.T) * (1 - s[bpt_t-1]**2)

        return [dU, dW, dV]

    # �����ݶ�
    def sgd_step(self, x, y, learning_rate):
        dU, dW, dV = self.bptt(x, y)

        self.U -= learning_rate * dU
        self.W -= learning_rate * dW
        self.V -= learning_rate * dV

    # ѵ��RNN
    def train(self, X_train, y_train, learning_rate=0.005, n_epoch=5):
        losses = []
        num_examples = 0

        for epoch in xrange(n_epoch):
            for i in xrange(len(y_train)):
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples += 1

            loss = self.loss(X_train, y_train)
            losses.append(loss)
            print 'epoch {0}: loss = {1}'.format(epoch+1, loss)
            # ����ʧ���ӣ�����ѧϰ��
            if len(losses) > 1 and losses[-1] > losses[-2]:
                learning_rate *= 0.5
                print 'decrease learning_rate to', learning_rate


unknown_token = 'UNKNOWN_TOKEN'
start_token = 'START_TOKEN'
end_token = 'END_TOKEN'

def generate_text(rnn, dict_words, index_of_words):
    # dict_words: type list; index_of_words: type dict
    sent = [index_of_words[start_token]]
    # Ԥ���´ʣ�֪�����ӵĽ���(END_TOKEN)
    while not sent[-1] == index_of_words[end_token]:
        next_probs, _ = rnn.forward(sent)
        sample_word = index_of_words[unknown_token]

        # ��Ԥ������ֲ����в������õ��µĴ�
        while sample_word == index_of_words[unknown_token]:
            samples = np.random.multinomial(1, next_probs[-1])
            sample_word = np.argmax(samples)
        # �������ɵ��к���Ĵ�(����ΪUNKNOWN_TOKEN�Ĵ�)�������
        sent.append(sample_word)

    new_sent = [dict_words[i] for i in sent[1:-1]]
    new_sent_str = ' '.join(new_sent)

    return new_sent_str

if __name__ == '__main__':
    file_path = r'../data/reddit-comments-2015-08.csv'
    dict_size = 8000
    myTokenFile = tokenFile.tokenFile2vector(file_path, dict_size)
    X_train, y_train, dict_words, index_of_words = myTokenFile.get_vector()
    rnn = myRNN(dict_size, hidden_dim=100, bptt_back=4)
    rnn.train(X_train[:200], y_train[:200], learning_rate=0.005, n_epoch=10)

    sent_str = generate_text(rnn, dict_words, index_of_words)
    print 'Generate sentence:', sent_str