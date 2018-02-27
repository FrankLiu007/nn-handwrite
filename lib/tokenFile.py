#!/usr/bin/env python
# -*- coding: gb18030 -*-
'''
Copyright (c) Baidu.com, Inc. All Rights Reserved

@Time     : 2018/2/27 15:16

@Author   : yangfengguang

@File     : tokenFile.py

@Software : PyCharm
'''
import numpy as np
import nltk, itertools, csv

TXTCODING = 'utf-8'
unknown_token = 'UNKNOWN_TOKEN'
start_token = 'START_TOKEN'
end_token = 'END_TOKEN'

# ���������ļ�Ϊ��ֵ����
class tokenFile2vector:
    def __init__(self, file_path, dict_size):
        self.file_path = file_path
        self.dict_size = dict_size

    # ���ı���ɾ��ӣ������Ͼ��ӿ�ʼ�ͽ�����־
    def _get_sentences(self):
        sents = []
        with open(self.file_path, 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            # ȥ����ͷ
            reader.next()
            # ����ÿ������Ϊ����
            sents = itertools.chain(*[nltk.sent_tokenize(x[0].decode(TXTCODING).lower()) for x in reader])
            sents = ['%s %s %s' % (start_token, sent, end_token) for sent in sents]
            print 'Get {} sentences.'.format(len(sents))

            return sents

    # �õ�ÿ�仰�ĵ��ʣ����õ��ֵ估�ֵ���ÿ���ʵ��±�
    def _get_dict_wordsIndex(self, sents):
        sent_words = [nltk.word_tokenize(sent) for sent in sents]
        word_freq = nltk.FreqDist(itertools.chain(*sent_words))
        print 'Get {} words.'.format(len(word_freq))

        common_words = word_freq.most_common(self.dict_size-1)
        # ���ɴʵ�
        dict_words = [word[0] for word in common_words]
        dict_words.append(unknown_token)
        # �õ�ÿ���ʵ��±꣬�������ɴ�����
        index_of_words = dict((word, ix) for ix, word in enumerate(dict_words))

        return sent_words, dict_words, index_of_words

    # �õ�ѵ������
    def get_vector(self):
        sents = self._get_sentences()
        sent_words, dict_words, index_of_words = self._get_dict_wordsIndex(sents)

        # ��ÿ��������û�������ʵ�dict_words�еĴ��滻Ϊunknown_token
        for i, words in enumerate(sent_words):
            sent_words[i] = [w if w in dict_words else unknown_token for w in words]

        X_train = np.array([[index_of_words[w] for w in sent[:-1]] for sent in sent_words])
        y_train = np.array([[index_of_words[w] for w in sent[1:]] for sent in sent_words])

        return X_train, y_train, dict_words, index_of_words