# -*- coding:utf-8 -*-
import numpy as np

word_model = {}

def load_word_model(class_file):
    """
    輸入詞向量檔案，生成word2vec對詞聚類模型
    :param class_file: 詞向量檔案
    :return: model: 詞聚類模型, dict { word: class }
    """
    with open(class_file, "r") as f:
        for line in f:
            w = line.strip().split()
            word_model[w[0]] = int(w[1])

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

def vectorize_single_news(dim, news_dict):
    id_matrix = np.eye(dim)

    news_id = news_dict['_id']
    news_stem = news_dict['stemContent'].split()

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

if __name__ == "__main__":
    load_word_model(class_file="2200.txt")
    a = {'_id':100, 'stemContent':"unit sta qwe kin good is aws kill job employ unit sta qwe kin good is aws kill job employ unit sta qwe kin good is aws kill job employ unit sta qwe kin good is aws kill job employ unit sta qwe kin good is aws kill job employ unit sta qwe kin good is aws kill job employ"}
    b = {'_id':1, 'stemContent':"unit sta qwe "}
    print type(vectorize_single_news(2200, b))