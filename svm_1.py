# SVM分類，詞頻統計，有卡方驗證和min_df降維
import jieba
import jieba.posseg as pseg
import os
import sys
import math
import json
from collections import OrderedDict

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
import numpy as np


class TF_IDF():
    def __init__(self):
        self.docs = {}
        self.get_seg_docs()
        self.stopword = []

        self.bow = {}

    def read_file(self, path, type):
        # file.read([size])从文件读取指定的字节数，如果未给定或为负则读取所有。
        if type == 'json':
            with open(path, 'r', encoding='utf-8') as file:
                data = json.loads(file.read())
        elif type == 'txt':
            with open(path, 'r', encoding='utf-8') as file:
                data = file.read()
        return data

    def get_seg_docs(self):
        self.train_seg_docs = []
        self.test_seg_docs = []
        FOLDER_NAME = 'data'
        # DOCUMENT = 'test.json'
        DOCUMENT = 'ettoday.news.json'
        STOPWORD = 'stopword.txt'
        # 其中__file__虽然是所在.py文件的完整路径，但是这个变量有时候返回相对路径，有时候返回绝对路径，因此还要用os.path.realpath()函数来处理一下。
        # 获取当前文件__file__的路径，    __file__是当前执行的文件
        FILE_DIR = os.path.join(os.path.split(
            os.path.realpath(__file__))[0], FOLDER_NAME)

        self.docs = self.read_file(FILE_DIR + '/' + DOCUMENT, 'json')
        self.stopword = self.read_file(FILE_DIR + '/' + STOPWORD, 'txt')
        # 新聞所有類別
        all_ca = ['ET車雲', '新奇', 'ET來了', '寵物動物', '遊戲', '健康', '房產雲', '論壇', '政治', '社會', '運勢', '國際',
                  '公益', '旅遊', '財經', '保險', '體育', '名家', '地方', '軍武', '男女', '3C', '法律', '時尚', '消費', '大陸', '影劇', '生活']

        train_news, test_news = train_test_split(
            self.docs, test_size=0.3, random_state=1)  # 分訓練集和測試集
        # self.train_seg_docs = []
        # self.test_seg_docs = []
        self.train_news_ca = []
        self.test_news_ca = []
        # self.ca = []
        # jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)，或者用
        # jieba.lcut 以及 jieba.lcut_for_search 直接返回 list
        # isalpha()去除不是字母組成的字，中文字也算，e.g:\r\n，不然會斷出\r\n

        # 訓練集
        for i in range(len(train_news)):
            # 計算幾個類別
            # self.ca.append(self.docs[i]['category'])
            ca = all_ca.index(train_news[i]['category'])  # 判斷新聞是哪個類別
            self.train_news_ca.append(ca)  # 新聞類別list

            content_str = ''
            # content_seg = []
            for w in jieba.lcut(train_news[i]['content']):
                if len(w) > 1 and w not in self.stopword and w.isalpha():
                    content_str = content_str+' '+w
                    # content_seg.append(w)
            self.train_seg_docs.append(content_str)
        # 測試集
        for i in range(len(test_news)):
                    # 計算幾個類別
                    # self.ca.append(self.docs[i]['category'])
            ca = all_ca.index(test_news[i]['category'])  # 判斷新聞是哪個類別
            self.test_news_ca.append(ca)  # 新聞類別list

            content_str = ''
            # content_seg = []
            for w in jieba.lcut(test_news[i]['content']):
                if len(w) > 1 and w not in self.stopword and w.isalpha():
                    content_str = content_str+' '+w
                    # content_seg.append(w)
            self.test_seg_docs.append(content_str)

        # print(self.train_seg_docs)
        # print(self.test_seg_docs)


if __name__ == '__main__':
    tf_idf = TF_IDF()
    train_corpus = tf_idf.train_seg_docs
    test_corpus = tf_idf.test_seg_docs
    train_news_ca = tf_idf.train_news_ca
    test_news_ca = tf_idf.test_news_ca

    # transformer = TfidfVectorizer()  # 该类会统计每个词语的tf-idf权值
    vectorizer = CountVectorizer(min_df=2)
    # transformer = TfidfVectorizer(max_features=200, min_df=2)
    # transformer = TfidfVectorizer(
    #     min_df=2, max_df=100)  # 如果詞只在一篇文檔中出現，就從文檔字典中去除
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    # 訓練集用fit_transform，測試集用transform，這樣訓練資料和測試資料就會有相同的關鍵字集合
    X_train = vectorizer.fit_transform(train_corpus)
    X_test = vectorizer.transform(test_corpus)

    # train_weight = train_tfidf.toarray()
    # test_weight = test_tfidf.toarray()

    ch2 = SelectKBest(chi2, k=500)
    x_train = ch2.fit_transform(X_train, train_news_ca)

    x_test = ch2.transform(X_test)

    clf = SVC()
    param_grid = {'kernel': ('linear', 'poly'), 'C': [0.1, 1]}

    grid_search = GridSearchCV(clf, param_grid=param_grid,
                               cv=5, scoring=make_scorer(accuracy_score))
    grid_search.fit(x_train, train_news_ca)
    y = grid_search.predict(x_test)
    print(grid_search.best_estimator_)
    print(accuracy_score(test_news_ca, y))
    print(classification_report(test_news_ca, y))
