# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import os
import re
from scipy.spatial import distance
import numpy as np
from nltk.stem.porter import PorterStemmer
from datetime import *
import time

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s " %(message)s', level=logging.INFO)

class Function():
    def __init__(self):
        self.word_model = {}

    def load_word_model(self, dim, class_file):
        """
        輸入詞向量檔案，生成word2vec對詞聚類模型
        :param class_file: 詞向量檔案
        :return: model: 詞聚類模型, dict { word: class }
        """
        with open(class_file, "r") as f:
            for line in f:
                w = line.strip().split()
                self.word_model[w[0]] = int(w[1])
                assert (dim >= int(w[1]))

        self.stopwords = []
        with open("utils\stopwords_en.txt", 'r') as f:
            for line in f:
                self.stopwords.append(line.strip())

        self.porter_stemmer = PorterStemmer()
        self.pattern = re.compile("[a-zA-Z]+")

    def cal_similarity(self, vec1, vec2):
        """
        計算兩向量間的cosine similarity
        :param vec1: 向量1
        :param vec2: 向量2
        :return: similarity: consine similarity
        """
        return 1.0 - distance.cosine(vec1, vec2)

    # get mean of square error
    def get_mse(self, vecs, centroid):
        rows, cols = vecs.shape
        # centroid = np.mean(vecs, axis=0)
        centroid_all = np.tile(centroid, (rows, 1))
        MSE = np.mean(np.square(vecs - centroid_all))
        return MSE

    def get_cos(self, vecs, centroid):
        rows, cols = vecs.shape
        # centroid = np.mean(vecs, axis=0)
        centroid_all = np.tile(centroid, (rows, 1))
        COS = np.mean(distance.cdist(vecs, centroid_all, 'cosine'))
        COS_STD = np.std(np.mean(distance.cdist(vecs, centroid_all, 'cosine'), axis=1))
        return COS, COS_STD

    def vectorize_single_news(self, dim, news_dict):
        id_matrix = np.eye(dim)

        news_id = news_dict['_id']
        news_stem = news_dict['stemContent'].split()
        news_lower = news_dict['lowerContent'].split()

        vector = np.zeros(dim)
        word_count = 0
        for word in news_stem:
            if word_count > 250:
                break
            try:
                vector += id_matrix[int(self.word_model[word])]
                word_count += 1
            except KeyError:
                pass

        if word_count != 0:
            vector /= word_count
        return vector

    def preprocess(self, s):
        # preprocess here
        tokens = s.strip().split()
        stem_content = ""
        lower_content = ""
        for token in tokens:
            match = re.match(self.pattern, token)
            # regular expression
            if match:
                # remove stop words
                if match.group().lower() not in self.stopwords:
                    # stemming
                    lower_token = match.group().lower()
                    lower_content += lower_token + " "
                    try:
                        stem = self.porter_stemmer.stem(lower_token)
                        stem_content += stem + " "
                    except:
                        pass
        return lower_content, stem_content

    def simple_content_abs(self, content):
        return ".".join(content.strip().split('.')[:3]) + "."

    def get_content_abs(self, dim, content, centroid, r=0.1):
        """
        MEAD方法, score = C + P + F
        :param dim: dimension
        :param content: str
        :param centroid: numpy array
        :param r: compression ratio
        :return: compress_content: str
        """
        # 分成多個句子
        id_matrix = np.eye(dim)
        content_split = content.split('.')
        min_fsent_len = 8

        c_val = np.zeros((len(content_split),))
        p_val = np.zeros_like(c_val)
        f_overlap = np.zeros_like(c_val)
        n = p_val.shape[0]

        # 如果取到的第一句太短, 擴展到前兩句
        if content_split[0] < min_fsent_len:
            first_sent = ".".join(content_split[:2])
            oth_sents = content_split[2:]
        else:
            first_sent = content_split[0]
            oth_sents = content_split[1:]

        first_sent_vec = np.zeros(dim)
        wd_count = 0
        _, first_sent_stem = self.preprocess(first_sent)
        for wd in first_sent_stem:
            try:
                first_sent_vec += id_matrix[int(self.word_model[wd])]
                wd_count += 1
            except:
                pass
        # if first sentence is not exist
        if not wd_count:
            return self.simple_content_abs(content)

        first_sent_vec /= wd_count
        c_val[0] = self.cal_similarity(first_sent_vec, centroid)
        p_val[0] = (n + 1.0) / n * c_val[0]
        f_overlap[0] = 1.0

        oth_sents_vecs = np.zeros((len(oth_sents), dim))
        for i, sents in enumerate(oth_sents):
            wd_count = 0
            _, oth_sent_stem = self.preprocess(sents)
            for wd in oth_sent_stem:
                try:
                    oth_sents_vecs[i, :] += id_matrix[int(self.word_model[wd])]
                    wd_count += 1
                except:
                    pass

            if wd_count:
                oth_sents_vecs[i, :] /= wd_count
                c_val[i+1] = self.cal_similarity(oth_sents_vecs[i, :], centroid)
                p_val[i+1] = float(n - i) / n * c_val[i+1]
                f_overlap[i+1] = self.cal_similarity(oth_sents_vecs[i, :], first_sent_vec)

        score = c_val + p_val + f_overlap
        sort_score = sorted([(i, j) for i, j in enumerate(score)], key=lambda x:x[1], reverse=True)
        compress_content = ".".join([content_split[i] for i, j in sort_score[: int(n*r)+1]]) + "."

        return compress_content

    def time2time_stamp(self, t):
        timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
        timeStamp = float(time.mktime(timeArray))
        return timeStamp

    def time_stamp2time(self, t):
        return str(datetime.fromtimestamp(t))

    def generate_timeinfo(self, start_t, end_t):
        # time_info = ((current_time_start_ts, current_time_start_t), (current_time_end_ts, current_time_end_t))
        start_ts = self.time2time_stamp(start_t)
        end_ts = self.time2time_stamp(end_t)
        return ((start_ts, start_t), (end_ts, end_t))

def test_function():
    print "test init function"
    function = Function()
    print "------------------"
    print "test preprocess"
    s = "Today is a beautiful day."
    lc, sc = function.preprocess(s)
    print lc, sc
    print "------------------"
    print "test stopwords"
    print function.stopwords
    print "------------------"

if __name__ == "__main__":
    test_function()