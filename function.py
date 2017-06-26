# -*- coding:utf-8 -*-
import numpy as np

def load_model(class_file):
    """
    輸入詞向量檔案，生成word2vec對詞聚類模型
    :param class_file: 詞向量檔案
    :return: model: 詞聚類模型, dict { word: class }
    """
    model = {}
    class_file = open(class_file, "r")
    for line in class_file.readlines():
        w = line.strip().split()
        model[w[0]] = int(w[1])
    return model

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
