# -*- coding:utf-8 -*-
import os
import sys
import json
import numpy as np
from header import *
from reader import *

class Clustering():
    def __init__(self, sim_thres, merge_sim_thres, subevent_sim_thres, dim, class_file, news_reader, event_reader):
        self.__dim = dim
        self.__sim_thres = sim_thres
        self.__merge_sim_thres = merge_sim_thres
        self.__subevent_sim_thres = subevent_sim_thres
        self.__mse_thres = 5 * 1e-6
        self.__word_model = self.__load_model(class_file=class_file)
        self.__events = {}
        self.__clusters_vec = {}
        self.__clusters_id = {}
        self.__centroids = {}
        self.__news_len = 30
        self.__news_count = 0
        self.__cluster_count = 0
        self.__event_count = 0
        self.__news_reader = news_reader
        self.__event_reader = event_reader
        self.__start = time.time()
        self.n_eid = ""
        self.__start_datetime = str(datetime.fromtimestamp(int(self.__start)))
        current_base = os.path.abspath('.')
        self.outbase = os.path.join(current_base, "Output",
                               's' + str(self.__sim_thres) + 'ms' + str(self.__merge_sim_thres) + 'sub' + str(
                                   self.__subevent_sim_thres) + 'dim' + str(self.__dim))

    # def update_count(self):
    #     self.__cluster_count += 1

    def __load_model(self, class_file):
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

    def vectorize(self, news_list):
        """
        輸入一段新聞，並利用新聞中的stemContent將文檔向量化
        目前使用方法為每個詞的權重都為1，生成向量將除以所有詞總數
        :param news_list: 一段新聞, list [ dict news_info { news.json }, ... , ]
        :return: vectors: 根據給定的dim維度生成的全部文檔向量, list [ tuple news ( _id, vector ), ... , ]
        """
        dim = self.__dim

        id_matrix = np.eye(dim)
        vectors = list()

        for news in news_list:
            news_id = news['_id']
            news_stem = news['stemContent'].split()

            if len(news_stem) > self.__news_len:
                vector = np.zeros(dim)
                word_count = 0
                for word in news_stem:
                    try:
                        vector += id_matrix[int(self.__word_model[word])]
                        word_count += 1
                    except KeyError:
                        pass

                if word_count != 0:
                    vector /= word_count
                vectors.append((news_id, vector))
                self.__news_count += 1

        return vectors

    def _cal_similarity(self, vec1, vec2):
        """
        計算兩向量間的cosine similarity
        :param vec1: 向量1
        :param vec2: 向量2
        :return: similarity: consine similarity
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _time_stamp2time(self, t):
        """
        轉換輸入的time_stamp為用於query的dateime格式
        :param t: time_stamp
        :return: datetime格式
        """
        return datetime.fromtimestamp(t)

    def online_clustering(self, vectors, sim_thres):
        """
        對輸入的vectors做online clustering聚類
        :param vectors: 全部文檔向量, list [ tuple news ( _id, vector ), ... , ]
        :param sim_thres: 相似度閾值
        :return: clusters: 向量聚類結果, list [ array cluster0 [ (vec0), ... , (vecN) ] , ... , ]
        :return: centroids: 向量聚類中心, list [ cluster0 (vec0), ... , clusterN (vecN) ]
        :return: clusters_id: 聚類新聞id, list [ list cluster0 [ (_id_0), ... , (_id_N) ], ... , ]
        """
        clusters_vec = {}
        clusters_id = {}
        centroids = {}

        for x in vectors:
            vid = x[0]
            vec = x[1]

            try:
                # 新聞計算最相似的聚類中心，並回傳最大相似值 ( cluster_id, sim )
                max_similarity = max([(key, self._cal_similarity(vec, centroids[key])) \
                                      for key in centroids], key=lambda t: t[1])
            except ValueError:
                max_similarity = (0, 0)

            bestmukey = max_similarity[0]

            # 最大相似度小於相似度閾值, 產生新事件
            if max_similarity[1] < sim_thres:
                key = self.n_eid + "E" + str(self.__event_count)
                clusters_vec[key] = np.array([vec])
                clusters_id[key] = [vid]
                centroids[key] = np.array(vec)
                self.__event_count += 1
            # 最大相似度大於相似度閾值
            else:
                clusters_vec[bestmukey] = np.vstack((clusters_vec[bestmukey], np.array(vec)))
                clusters_id[bestmukey].append(vid)
                centroids[bestmukey] = np.mean(clusters_vec[bestmukey], axis=0)

        return clusters_vec, clusters_id, centroids

    def read_events(self, current_time):
        result = self.__event_reader.query_recent_events(current_time)
        self.__clusters_vec = {}
        self.__clusters_id = {}
        self.__centroids = {}

        event_count = 0
        # news_all = []
        for event in result:
            event_id = event['id']
            self.__events[event_id] = event

            news_in_event = []
            news_id_in_event = []
            for news in event['articles']:
                news_id = news['id']
                queried_news = self.__news_reader.query_mongoDB_by_item({'_id':news_id})
                for i in queried_news:
                    news_in_event.append(i)
                news_id_in_event.append(news_id)

            self.__clusters_vec[event_id] = self.vectorize(news_in_event)
            self.__clusters_id[event_id] = news_id_in_event
            self.__centroids[event_id] = np.mean(self.__clusters_vec[event_id], axis=0)
            event_count += 1
            # news_all.append(news_in_event)
        return event_count

    def online_clustering_merge(self, current_time, result):
        """
        將完成聚類的新聞合併到原有的事件中
        :param clusters_vec: dict
        :param clusters_id: dict
        :param centroids: dict
        :return:
        """
        self.n_eid = self.__event_reader.create_event_id(current_time=current_time)
        clusters_vec, clusters_id, centroids = result
        if not self.__clusters_vec and not self.__clusters_id and not self.__centroids:
            self.__clusters_vec = clusters_vec
            self.__clusters_id = clusters_id
            self.__centroids = centroids
        else:
            for event_id in centroids:
                cluster_vec = clusters_vec[event_id]
                cluster_id = clusters_id[event_id]
                centroid = centroids[event_id]

                max_similarity = max([(eid, self._cal_similarity(centroid, self.__centroids[eid])) \
                                          for eid in self.__centroids], key=lambda t: t[1])

                bestmukey = max_similarity[0]

                # only update cluster, cluster_id
                if max_similarity[1] < self.__merge_sim_thres:
                    eid_new = self.n_eid + "E" + str(self.__event_count)    # 20170620170000E0
                    self.__clusters_vec[eid_new] = np.array(cluster_vec)
                    self.__clusters_id[eid_new] = cluster_id
                    self.__event_count += 1
                else:
                    self.__clusters_vec[bestmukey] = np.vstack((self.__clusters_vec[bestmukey], cluster_vec))
                    self.__clusters_id[bestmukey].extend(cluster_id)

    # get mean of square error
    def get_cluster_mse(self, vecs):
        rows, cols = vecs.shape
        centroid = np.mean(vecs, axis=0)
        centroid_all = np.tile(centroid, (rows, 1))
        MSE = np.mean(np.square(vecs - centroid_all))
        return MSE

    # input = (cluster_id, [vecs]) // cluster info
    def split_cluster(self, cluster):
        event_id = cluster[0]
        event_vecs = cluster[1]
        # print event_id, event_vecs
        clusters_id = self.__clusters_id[event_id]

        vectors = [ (clusters_id[i[0]], i[1]) for i in enumerate(event_vecs) ]
        n_clusters_vec, n_clusters_id, n_centroids = self.online_clustering(vectors=vectors, sim_thres=self.__subevent_sim_thres)

        # replace the original cluster info with 1st newly generated cluster info
        self.__clusters_vec[event_id] = n_clusters_vec[0]
        self.__clusters_id[event_id] = n_clusters_id[0]
        self.__centroids[event_id] = n_centroids[0]

        outbase = self.outbase
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        outbase = os.path.join(outbase, "SplitResult")
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        outbase = os.path.join(outbase, "Split"+str(self.__cluster_count))
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        out_file = os.path.join(outbase, "split_event" + str(event_id))
        out = open(out_file, "w")
        out.write("Event " + str(event_id) + "\n")
        for key in n_clusters_id:
            news_id_all = n_clusters_id[key]
            out.write("Cluster " + str(key) + " num = " + str(len(news_id_all)) + "\n")
            for news_id in news_id_all:
                result = self.__news_reader.query_mongoDB_by_item({"_id":news_id})
                for news in result:
                    out.write("Title: " + news['title'] + " Time: " + news['crawlTime'] + " Content: " + news['content'] + "\n")
        out.close()

        n_clusters_vec.pop(0)
        n_clusters_id.pop(0)
        n_centroids.pop(0)

        return (n_clusters_vec, n_clusters_id, n_centroids)

    def reevalute_centroids(self):
        # self.__centroids = [ np.zeros(self.__dim) for x in xrange(len(self.__clusters_vec)) ]
        self.__centroids = { key : np.zeros(self.__dim) for key in self.__clusters_vec }
        clusters_vec = {}
        clusters_id = {}
        centroids = {}

        # mse_out = open(os.path.join(outbase, "mse"+str(self.__cluster_count)), "w")
        for event_id in self.__clusters_vec:
            vecs = self.__clusters_vec[event_id]
            self.__centroids[event_id] += np.mean(vecs, axis=0)
            if len(vecs) > 1:
                mse = self.get_cluster_mse(vecs)
                # mse_out.write("Cluster " + str(cid) + "\t" + str(mse) + "\n")
                if mse > self.__mse_thres:
                    cluster = (event_id, vecs)
                    n_clusters_vec, n_clusters_id, n_centroids = self.split_cluster(cluster=cluster)
                    clusters_vec.update(n_clusters_vec)
                    clusters_id.update(n_clusters_id)
                    centroids.update(n_centroids)
        # mse_out.close()
        return clusters_vec, clusters_id, centroids

    # def merge_split_event(self, result):
    #     clusters_vec, clusters_id, centroids = result
    #     n_clusters_vec = {}
    #     n_clusters_id = {}
    #     n_centroids = {}
    #
    #     for key in clusters_vec.iterkeys():
    #         eid_new = self.n_eid + "E" + str(self.__event_count)  # 20170620170000E0
    #         n_clusters_vec[eid_new] = clusters_vec[key]
    #         n_clusters_id[eid_new] = clusters_id[key]
    #         n_centroids[eid_new] = centroids[key]
    #         self.__event_count += 1
    #
    #     self.__clusters_vec.update(n_clusters_vec)
    #     self.__clusters_id.update(n_clusters_id)
    #     self.__centroids.update(n_centroids)

    def rearrange_cluster(self):
        event_id = sorted(((i, self.__clusters_id[i]) for i in xrange(len(self.__clusters_id))), key=lambda v:len(v[1]), reverse=True)

        clusters_vec = []
        clusters_id = []
        centroids = []
        for i in event_id:
            sort_id = i[0]
            clusters_vec.append(self.__clusters_vec[sort_id])
            clusters_id.append(self.__clusters_id[sort_id])
            centroids.append(self.__centroids[sort_id])

        self.__clusters_vec = clusters_vec
        self.__clusters_id = clusters_id
        self.__centroids = centroids

    def clustering_news(self, news_list):
        print "vectorize"
        vectors = self.vectorize(news_list=news_list)
        print "time = ", time.time() - self.__start
        print

        print "clustering"
        clusters_vec, clusters_id, centroids = self.online_clustering(vectors=vectors, sim_thres=self.__sim_thres)
        print "cluster = ", len(clusters_id)
        print "time = ", time.time() - self.__start
        print

        return (clusters_vec, clusters_id, centroids)

    def merge_events(self, current_time, result):
        # clusters_vec, clusters_id, centroids = result
        print "read events"
        event_num = self.read_events(current_time=current_time)
        print "total event = ", event_num
        print "time = ", time.time() - self.__start
        print

        print "merge"
        print "previous cluster = ", len(self.__clusters_id)
        self.online_clustering_merge(current_time=current_time, result=result)
        print "merged cluster = ", len(self.__clusters_id)
        print "time = ", time.time() - self.__start
        print

    def reevaluate(self, current_time):
        print "re-evaluate centroids"
        print "previous cluster = ", len(self.__clusters_id)
        result = self.reevalute_centroids()
        print "time = ", time.time() - self.__start
        print

        print "merge split event"
        self.online_clustering_merge(current_time=current_time, result=result)
        print "re-evaluated cluster = ", len(self.__clusters_id)
        print "time = ", time.time() - self.__start
        print "--------------------------------------------"

        # print "re-arange clusters"
        # self.rearrange_cluster()
        # print "time = ", time.time() - self.__start
        # print

    def write_news(self):
        outbase = self.outbase
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        single_count = 0
        total_news_count = 0
        out = open(os.path.join(outbase, str(self.n_eid)), "w")
        id_out = open(os.path.join(outbase, "cid"), "w")
        for key in self.__clusters_id:
            cluster = self.__clusters_id[key]
            total_news_count += len(cluster)
            out.write("Cluster " + str(key) + " num = " + str(len(cluster)) + "\n")
            id_out.write(json.dumps(cluster) + "\n")
            if len(cluster) == 1:
                single_count += 1
            for news_id in cluster:
                result = self.__news_reader.query_mongoDB_by_item({"_id":news_id})
                for news in result:
                    out.write("Title: " + news['title'] + " Time: " + news['crawlTime'] + " Content: " + news['content'] + "\n")
        out.close()
        id_out.close()

        self.write_paras(single_count=single_count, cluster_num=len(self.__clusters_id))

    def write_paras(self, single_count, cluster_num):
        outbase = self.outbase
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        paras = open(os.path.join(outbase, "paras"+str(self.__cluster_count)), "w")
        paras.write("Time = " + str(time.time() - self.__start) + "\n")
        paras.write("Similarity = " + str(self.__sim_thres) + "\n")
        paras.write("Merge similarity = " + str(self.__merge_sim_thres) + "\n")
        paras.write("Subevent similarity = " + str(self.__subevent_sim_thres) + "\n")
        paras.write("Mse thres = " + str(self.__mse_thres) + "\n")
        paras.write("Total news num = " + str(self.__news_count) + "\n")
        paras.write("Single event = " + str(single_count) + "\n")
        paras.write("Clusters = " + str(cluster_num) + "\n")
        paras.close()

        # self.update_count()