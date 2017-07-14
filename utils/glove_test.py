# -*- coding:utf-8 -*-
import re
import numpy as np
from gensim.models import KeyedVectors
from reader import NewsReader, EventReader
from nltk.stem.porter import PorterStemmer

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s " %(message)s', level=logging.INFO)

word_model = {}
word_vectors = KeyedVectors.load_word2vec_format("model/glove_w2v.6B.200d.txt")
stopwords = []

porter_stemmer = PorterStemmer()
pattern = re.compile("[a-zA-Z]+")

def init():
    sw_file = open("stopwords_en.txt", 'r')
    with sw_file as f:
        for line in f:
            stopwords.append(line.strip())

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