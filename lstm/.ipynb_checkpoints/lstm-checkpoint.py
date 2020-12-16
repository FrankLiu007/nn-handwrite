#!/usr/bin/env python


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

# 输出单元激活函数
def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

class myLSTM:
    def __init__(self, data_dim, hidden_dim=100):
        # data_dim: 词向量维度，即词典长度; hidden_dim: 隐单元维度
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        # 初始化权重向量
        self.whi, self.wxi, self.bi = self._init_wh_wx()
        self.whf, self.wxf, self.bf = self._init_wh_wx()
        self.who, self.wxo, self.bo = self._init_wh_wx()
        self.wha, self.wxa, self.ba = self._init_wh_wx()
        self.wy, self.by = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                   (self.data_dim, self.hidden_dim)), \
                           np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                   (self.data_dim, 1))

    # 初始化 wh, wx, b
    def _init_wh_wx(self):
        wh = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                   (self.hidden_dim, self.hidden_dim))
        wx = np.random.uniform(-np.sqrt(1.0/self.data_dim), np.sqrt(1.0/self.data_dim),
                                   (self.hidden_dim, self.data_dim))
        b = np.random.uniform(-np.sqrt(1.0/self.data_dim), np.sqrt(1.0/self.data_dim),
                                   (self.hidden_dim, 1))

        return wh, wx, b

    # 初始化各个状态向量
    def _init_s(self, T):
        iss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # input gate
        fss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # forget gate
        oss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # output gate
        ass = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # current inputstate
        hss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # hidden state
        css = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # cell state
        ys = np.array([np.zeros((self.data_dim, 1))] * T)    # output value

        return {'iss': iss, 'fss': fss, 'oss': oss,
                'ass': ass, 'hss': hss, 'css': css,
                'ys': ys}

    # 前向传播，单个x
    def forward(self, x):
        # 向量时间长度
        T = len(x)
        # 初始化各个状态向量
        stats = self._init_s(T)

        for t in range(T):
            # 前一时刻隐藏状态
            ht_pre = np.array(stats['hss'][t-1]).reshape(-1, 1)

            # input gate
            stats['iss'][t] = self._cal_gate(self.whi, self.wxi, self.bi, ht_pre, x[t], sigmoid)
            # forget gate
            stats['fss'][t] = self._cal_gate(self.whf, self.wxf, self.bf, ht_pre, x[t], sigmoid)
            # output gate
            stats['oss'][t] = self._cal_gate(self.who, self.wxo, self.bo, ht_pre, x[t], sigmoid)
            # current inputstate
            stats['ass'][t] = self._cal_gate(self.wha, self.wxa, self.ba, ht_pre, x[t], tanh)

            # cell state, ct = ft * ct_pre + it * at
            stats['css'][t] = stats['fss'][t] * stats['css'][t-1] + stats['iss'][t] * stats['ass'][t]
            # hidden state, ht = ot * tanh(ct)
            stats['hss'][t] = stats['oss'][t] * tanh(stats['css'][t])

            # output value, yt = softmax(self.wy.dot(ht) + self.by)
            stats['ys'][t] = softmax(self.wy.dot(stats['hss'][t]) + self.by)

        return stats

    # 计算各个门的输出
    def _cal_gate(self, wh, wx, b, ht_pre, x, activation):
        return activation(wh.dot(ht_pre) + wx[:, x].reshape(-1,1) + b)

    # 预测输出，单个x
    def predict(self, x):
        stats = self.forward(x)
        pre_y = np.argmax(stats['ys'].reshape(len(x), -1), axis=1)
        return pre_y

    # 计算损失， softmax交叉熵损失函数， (x,y)为多个样本
    def loss(self, x, y):
        cost = 0
        for i in xrange(len(y)):
            stats = self.forward(x[i])
            # 取出 y[i] 中每一时刻对应的预测值
            pre_yi = stats['ys'][xrange(len(y[i])), y[i]]
            cost -= np.sum(np.log(pre_yi))

        # 统计所有y中词的个数, 计算平均损失
        N = np.sum([len(yi) for yi in y])
        ave_loss = cost / N

        return ave_loss

     # 初始化偏导数 dwh, dwx, db
    def _init_wh_wx_grad(self):
        dwh = np.zeros(self.whi.shape)
        dwx = np.zeros(self.wxi.shape)
        db = np.zeros(self.bi.shape)

        return dwh, dwx, db

    # 求梯度, (x,y)为一个样本
    def bptt(self, x, y):
        dwhi, dwxi, dbi = self._init_wh_wx_grad()
        dwhf, dwxf, dbf = self._init_wh_wx_grad()
        dwho, dwxo, dbo = self._init_wh_wx_grad()
        dwha, dwxa, dba = self._init_wh_wx_grad()
        dwy, dby = np.zeros(self.wy.shape), np.zeros(self.by.shape)

        # 初始化 delta_ct，因为后向传播过程中，此值需要累加
        delta_ct = np.zeros((self.hidden_dim, 1))

        # 前向计算
        stats = self.forward(x)
        # 目标函数对输出 y 的偏导数
        delta_o = stats['ys']
        delta_o[np.arange(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:
            # 输出层wy, by的偏导数，由于所有时刻的输出共享输出权值矩阵，故所有时刻累加
            dwy += delta_o[t].dot(stats['hss'][t].reshape(1, -1))
            dby += delta_o[t]

            # 目标函数对隐藏状态的偏导数
            delta_ht = self.wy.T.dot(delta_o[t])

            # 各个门及状态单元的偏导数
            delta_ot = delta_ht * tanh(stats['css'][t])
            delta_ct += delta_ht * stats['oss'][t] * (1-tanh(stats['css'][t])**2)
            delta_it = delta_ct * stats['ass'][t]
            delta_ft = delta_ct * stats['css'][t-1]
            delta_at = delta_ct * stats['iss'][t]

            delta_at_net = delta_at * (1-stats['ass'][t]**2)
            delta_it_net = delta_it * stats['iss'][t] * (1-stats['iss'][t])
            delta_ft_net = delta_ft * stats['fss'][t] * (1-stats['fss'][t])
            delta_ot_net = delta_ot * stats['oss'][t] * (1-stats['oss'][t])

            # 更新各权重矩阵的偏导数，由于所有时刻共享权值，故所有时刻累加
            dwhf, dwxf, dbf = self._cal_grad_delta(dwhf, dwxf, dbf, delta_ft_net, stats['hss'][t-1], x[t])
            dwhi, dwxi, dbi = self._cal_grad_delta(dwhi, dwxi, dbi, delta_it_net, stats['hss'][t-1], x[t])
            dwha, dwxa, dba = self._cal_grad_delta(dwha, dwxa, dba, delta_at_net, stats['hss'][t-1], x[t])
            dwho, dwxo, dbo = self._cal_grad_delta(dwho, dwxo, dbo, delta_ot_net, stats['hss'][t-1], x[t])

        return [dwhf, dwxf, dbf,
                dwhi, dwxi, dbi,
                dwha, dwxa, dba,
                dwho, dwxo, dbo,
                dwy, dby]

    # 更新各权重矩阵的偏导数
    def _cal_grad_delta(self, dwh, dwx, db, delta_net, ht_pre, x):
        dwh += delta_net * ht_pre
        dwx += delta_net * x
        db += delta_net

        return dwh, dwx, db

    # 计算梯度, (x,y)为一个样本
    def sgd_step(self, x, y, learning_rate):
        dwhf, dwxf, dbf, \
        dwhi, dwxi, dbi, \
        dwha, dwxa, dba, \
        dwho, dwxo, dbo, \
        dwy, dby = self.bptt(x, y)

        # 更新权重矩阵
        self.whf, self.wxf, self.bf = self._update_wh_wx(learning_rate, self.whf, self.wxf, self.bf, dwhf, dwxf, dbf)
        self.whi, self.wxi, self.bi = self._update_wh_wx(learning_rate, self.whi, self.wxi, self.bi, dwhi, dwxi, dbi)
        self.wha, self.wxa, self.ba = self._update_wh_wx(learning_rate, self.wha, self.wxa, self.ba, dwha, dwxa, dba)
        self.who, self.wxo, self.bo = self._update_wh_wx(learning_rate, self.who, self.wxo, self.bo, dwho, dwxo, dbo)

        self.wy, self.by = self.wy - learning_rate * dwy, self.by - learning_rate * dby

    # 更新权重矩阵
    def _update_wh_wx(self, learning_rate, wh, wx, b, dwh, dwx, db):
        wh -= learning_rate * dwh
        wx -= learning_rate * dwx
        b -= learning_rate * db

        return wh, wx, b

    # 训练 LSTM
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
            if len(losses) > 1 and losses[-1] > losses[-2]:
                learning_rate *= 0.5
                print 'decrease learning_rate to', learning_rate

if __name__ == '__main__':
    # 获取数据
    file_path = r'../data/reddit-comments-2015-08.csv'
    dict_size = 8000
    myTokenFile = tokenFile.tokenFile2vector(file_path, dict_size)
    X_train, y_train, dict_words, index_of_words = myTokenFile.get_vector()

    # 训练LSTM
    lstm = myLSTM(dict_size, hidden_dim=100)
    lstm.train(X_train[:200], y_train[:200],
              learning_rate=0.005,
              n_epoch=3)