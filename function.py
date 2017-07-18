# -*- coding:utf-8 -*-
import re
import numpy as np
# from gensim.models import Word2Vec, KeyedVectors
from reader import NewsReader, EventReader
from nltk.stem.porter import PorterStemmer

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s " %(message)s', level=logging.INFO)

word_model = {}
word_vectors = {}
# word_vectors = KeyedVectors.load_word2vec_format("model/glove_w2v.6B.200d.txt")
# word_vectors = KeyedVectors.load("model/glove.6B.200d.txt")
sw_file = open("stopwords_en.txt", 'r')
stopwords = list()
with sw_file as f:
    for line in f:
        stopwords.append(line.strip())

porter_stemmer = PorterStemmer()
pattern = re.compile("[a-zA-Z]+")

def load_word_model(dim, class_file):
    """
    輸入詞向量檔案，生成word2vec對詞聚類模型
    :param class_file: 詞向量檔案
    :return: model: 詞聚類模型, dict { word: class }
    """
    with open(class_file, "r") as f:
        for line in f:
            w = line.strip().split()
            word_model[w[0]] = int(w[1])
            assert (dim >= int(w[1]))

def cal_similarity(vec1, vec2):
    """
    計算兩向量間的cosine similarity
    :param vec1: 向量1
    :param vec2: 向量2
    :return: similarity: consine similarity
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# get mean of square error
def get_mse(vecs):
    rows, cols = vecs.shape
    centroid = np.mean(vecs, axis=0)
    centroid_all = np.tile(centroid, (rows, 1))
    MSE = np.mean(np.square(vecs - centroid_all))
    return MSE

def get_dvi(centroids):
    dist = [ np.sqrt(np.sum(np.square(cvec1 - cvec2))) \
                 for eid2, cvec2 in centroids.iteritems() \
                 for eid1, cvec1 in centroids.iteritems() ]

    n_d = [sorted([ np.sqrt(np.sum(np.square(cvec1 - cvec2))) for eid2, cvec2 in centroids.iteritems() ]) for eid1, cvec1 in centroids.iteritems() ]
    dvi = sorted([ ( x[1][1] / x[-1][1]) for x in n_d ])[:15]
    # sort_dist = sorted(dist)
    # min_dist = sort_dist[len(centroids)]
    # max_dist = sort_dist[-1]
    #
    print dvi
    # return min_dist / max_dist

def vectorize_single_news(dim, news_dict):
    id_matrix = np.eye(dim)

    news_id = news_dict['_id']
    news_stem = news_dict['stemContent'].split()
    news_lower = news_dict['lowerContent'].split()

    vector = np.zeros(dim)
    word_count = 0
    for word in news_stem:
        try:
            vector += id_matrix[int(word_model[word])]
            word_count += 1
        except KeyError:
            pass

    if word_count != 0:
        vector /= word_count
    return vector

def vectorize_with_dis(dim, news_dict):
    # id_matrix = np.eye(dim)

    news_id = news_dict['_id']
    news_stem = news_dict['stemContent'].split()
    news_lower = news_dict['lowerContent'].split()

    vector = np.zeros(dim)
    word_count = 0
    for word in news_lower:
        try:
            vector += word_vectors[word]
            word_count += 1
        except KeyError:
            pass

    if word_count != 0:
        vector /= word_count
    return vector

def preprocess(s):
    # preprocess here
    tokens = s.strip().split()
    stem_content = ""
    lower_content = ""
    for token in tokens:
        match = re.match(pattern, token)
        # regular expression
        if match:
            # remove stop words
            if match.group().lower() not in stopwords:
                # stemming
                lower_token = match.group().lower()
                lower_content += lower_token + " "
                try:
                    stem = porter_stemmer.stem(lower_token)
                    stem_content += stem + " "
                except:
                    pass
    return lower_content, stem_content

def get_content_abs(dim, content, centroid, r):
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
    c_val = np.zeros((len(content_split),))
    p_val = np.zeros_like(c_val)
    f_overlap = np.zeros_like(c_val)
    n = p_val.shape[0]

    first_sent = content_split[0]
    oth_sents = content_split[1:]

    first_sent_vec = np.zeros(dim)
    wd_count = 0
    _, first_sent_stem = preprocess(first_sent)
    for wd in first_sent_stem:
        try:

            first_sent_vec += id_matrix[int(word_model[wd])]
            wd_count += 1
        except:
            pass
    if wd_count:
        first_sent_vec /= wd_count
        c_val[0] = cal_similarity(first_sent_vec, centroid)
        p_val[0] = (n + 1.0) / n * c_val[0]
        f_overlap[0] = 1.0

    oth_sents_vecs = np.zeros((len(oth_sents), dim))
    for i, sents in enumerate(oth_sents):
        wd_count = 0
        _, oth_sent_stem = preprocess(sents)
        for wd in oth_sent_stem:
            try:
                oth_sents_vecs[i, :] += id_matrix[int(word_model[wd])]
                wd_count += 1
            except:
                pass
        if wd_count:
            oth_sents_vecs[i, :] /= wd_count
            c_val[i+1] = cal_similarity(oth_sents_vecs[i, :], centroid)
            p_val[i+1] = float(n - i) / n * c_val[i+1]
            f_overlap[i+1] = cal_similarity(oth_sents_vecs[i, :], first_sent_vec)

    score = c_val + p_val + f_overlap
    sort_score = sorted([(i, j) for i, j in enumerate(score)], key=lambda x:x[1], reverse=True)
    compress_content = ".".join([content_split[i] for i, j in sort_score[: int(n*r)+1]])

    return compress_content

if __name__ == "__main__":
    IP_PORT = "10.1.1.46:27017"
    # load_word_model(dim=2200, class_file="2200.txt")
    # news_reader = NewsReader(uri=IP_PORT)
    # start_time_t = "2016-07-25 16:00:00"
    # end_time_t = "2016-07-26 18:00:00"
    # news_list = news_reader.query_many_by_time(start_time=start_time_t, end_time=end_time_t)
    a = {'_id':100, 'stemContent':"unit sta qwe kin good. is aws kill job employ unit sta qwe kin good is. aws kill job employ unit sta qwe kin. good is aws kill job employ unit. sta qwe kin good is aws kill. job employ unit sta qwe kin good. is aws kill job employ unit sta qwe kin good is aws kill. job employ."}
    b = {'_id':1, 'stemContent':"unit sta qwe "}
    # print type(vectorize_single_news(2200, b))
    get_content_abs(2200, a['stemContent'], np.random.rand((2200)), 0.2)