# -*- coding:utf-8 -*-
import os
import numpy as np
from header import get_event_json
from time import sleep
from tqdm import tqdm
from reader import *
from function import load_word_model, cal_similarity, get_mse, vectorize_single_news, get_content_abs

class Model():
    def __init__(self, sim_thres, merge_sim_thres, subevent_sim_thres, dim, class_file, news_reader, event_reader):
        load_word_model(dim=dim, class_file=class_file)
        self.__dim = dim
        self.__sim_thres = sim_thres
        self.__merge_sim_thres = merge_sim_thres
        self.__subevent_sim_thres = subevent_sim_thres
        self.__mse_thres = 1e-5
        self.__news = {}
        self.__events = {}
        self.__clusters_vec = {}
        self.__clusters_id = {}
        self.__centroids = {}
        self.__son2father_event = {} # single id: str
        self.__father2son_event = {} # son set: set of str
        self.mse = []
        self.__min_news_len = 50
        self.__news_count = 0
        self.__cluster_count = 0
        self.__event_count = 0
        self.__single_count = 0
        self.__news_reader = news_reader
        self.__event_reader = event_reader
        self.__start = time.time()
        self.__date = ""
        current_base = os.path.abspath('.')
        self.outbase = os.path.join(current_base, "Output",
                               's' + str(self.__sim_thres) + 'ms' + str(self.__merge_sim_thres) + 'sub' + str(
                                   self.__subevent_sim_thres) + 'dim' + str(self.__dim))

    def vectorize_mongolist(self, news_list):
        """
        輸入一段新聞，並利用新聞中的stemContent將文檔向量化
        目前使用方法為每個詞的權重都為1，生成向量將除以所有詞總數
        :param news_list: 一段新聞, list [ dict news_info { news.json }, ... , ]
        :return: vectors: 根據給定的dim維度生成的全部文檔向量, list [ tuple news ( _id, vector ), ... , ]
        """
        self.__news_count = news_list.count()
        vectors = list()
        # 進度條
        time.sleep(0.3)
        pbar = tqdm(total=self.__news_count, mininterval=0.5)
        pb = 0
        for news_dict in news_list:
            news_id = news_dict['_id']
            self.__news[news_id] = news_dict
            news_stem_content = news_dict['stemContent']
            news_lower_content = news_dict['lowerContent']
            news_len = len(news_stem_content)

            if news_len > self.__min_news_len:
                vector = vectorize_single_news(dim=self.__dim, news_dict=news_dict)
                # vector = vectorize_with_dis(dim=self.__dim, news_dict=news_dict)
                vectors.append((news_id, vector))
            pb += 1
            pbar.update(1)
            # if pb % 10 == 0:
            #     pbar.update(10)
        pbar.close()
        time.sleep(0.3)
        return vectors

    def online_clustering(self, vectors, sim_thres, mode="clustering", father_event_id=""):
        """
        對輸入的vectors做online clustering聚類
        :param vectors: 全部文檔向量, list [ tuple news ( _id, vector ), ... , ]
        :param sim_thres: 相似度閾值
        :return: clusters: 向量聚類結果, dict [ array cluster0 [ (vec0), ... , (vecN) ] , ... , ]
        :return: centroids: 向量聚類中心, dict [ cluster0 (vec0), ... , clusterN (vecN) ]
        :return: clusters_id: 聚類新聞id, dict [ list cluster0 [ (_id_0), ... , (_id_N) ], ... , ]
        """
        clusters_vec = {}
        clusters_id = {}
        centroids = {}

        time.sleep(0.3)
        pbar = 0

        # 判斷mode, 分為split re-clustering跟clustering
        has_father_event = False
        if mode == "split":
            has_father_event = True
        elif mode == "clustering":
            pbar = tqdm(total=len(vectors), mininterval=0.5)

        for x in vectors:
            vid = x[0]
            vec = x[1]

            try:
                # 新聞計算最相似的聚類中心，並回傳最大相似值 ( cluster_id, sim )
                max_similarity = max([(key, cal_similarity(vec, centroids[key])) \
                                      for key in centroids], key=lambda t: t[1])
            except:
                max_similarity = (0, 0)

            bestmukey = max_similarity[0]

            # 最大相似度小於相似度閾值, 產生新事件
            if max_similarity[1] < sim_thres:
                # 父事件
                if has_father_event:
                    key = father_event_id
                    has_father_event = False
                # 子事件
                else:
                    key = self.__date + "E" + str(self.__event_count)
                clusters_vec[key] = np.array([vec])
                clusters_id[key] = [vid]
                centroids[key] = np.array(vec)
                self.__event_count += 1

                if father_event_id:
                    self.__son2father_event[key] = father_event_id
                    if father_event_id in self.__father2son_event:
                        self.__father2son_event[father_event_id].add(key)
                    else:
                        key_list = [key]
                        self.__father2son_event[father_event_id] = set(key_list)
            # 最大相似度大於相似度閾值
            else:
                clusters_vec[bestmukey] = np.vstack((clusters_vec[bestmukey], np.array(vec)))
                clusters_id[bestmukey].append(vid)
                centroids[bestmukey] = np.mean(clusters_vec[bestmukey], axis=0)

            if mode == 'clustering':
                pbar.update(1)

        if mode == 'clustering':
            pbar.close()
        time.sleep(0.3)
        return clusters_vec, clusters_id, centroids

    def read_events(self, t):
        """
        從mongoDB event collection 讀取上一個階段聚類完成的event, 讀取後將event放入 self.__events 存儲
        在當中也會讀取mongoDB news collection, 並對於讀取的news作文檔向量化
        :param t: time string
        :return: event count: int
        """
        result = self.__event_reader.query_recent_events_by_time(t=t)

        time.sleep(0.3)
        pbar = tqdm(total=result.count(), mininterval=0.5)
        event_count = 0
        for event in result:
            event_id = event['_id']
            self.__events[event_id] = event

            news_vec_in_event = []
            news_id_in_event = []

            # 宣告在event_json裡面的news
            for news_in_event in event['articles']:
                news_id = news_in_event['id']
                # 讀取mongoDB news collection, 並作文檔向量化
                result = self.__news_reader.query_one_by_item({'_id':news_id})
                if result:
                    news_stem_content = result['stemContent']
                    news_lower_content = result['lowerContent']
                    news_content_len = len(news_stem_content)
                    if news_content_len > self.__min_news_len:
                        news_vec_in_event.append(vectorize_single_news(dim=self.__dim, news_dict=result))
                        # news_vec_in_event.append(vectorize_with_dis(dim=self.__dim, news_dict=result))
                        news_id_in_event.append(news_id)
                        self.__news[news_id] = result

            # 讀取event_jon中的層次關係
            childrens = event['childrens']
            father = event['father']
            if childrens:
                self.__father2son_event[event_id] = set(childrens)
            if father != -1:
                self.__son2father_event[event_id] = father

            # 將讀取的event放入全部聚類的存儲 self.__cluster_vec, self.__cluster_id, self.__centroids
            self.__clusters_vec[event_id] = np.array(news_vec_in_event)
            self.__clusters_id[event_id] = news_id_in_event
            self.__centroids[event_id] = np.mean(self.__clusters_vec[event_id], axis=0)
            event_count += 1
            pbar.update(1)
        pbar.close()
        time.sleep(0.3)
        return event_count

    def online_clustering_merge(self, cluster_triple):
        """
        將完成聚類的新聞合併到原有的事件中
        目前方法: 比較新舊event的cosine相似度, 在對相似度最大的event進行合併(合併閾值 default = 0.7)
        :param clusters_vec: dict
        :param clusters_id: dict
        :param centroids: dict
        :return:
        """
        clusters_vec, clusters_id, centroids = cluster_triple
        # 沒有讀取到event
        if not self.__clusters_vec and not self.__clusters_id and not self.__centroids:
            self.__clusters_vec = clusters_vec
            self.__clusters_id = clusters_id
            self.__centroids = centroids
        # 讀取到event
        else:
            time.sleep(0.3)
            pbar = tqdm(total=len(centroids), mininterval=0.5)
            # 遍歷所有新生成的event, 對讀取的舊event評估進行合併
            for event_id in centroids:
                cluster_vec = clusters_vec[event_id]
                cluster_id = clusters_id[event_id]
                centroid = centroids[event_id]

                max_similarity = max([(eid, cal_similarity(centroid, self.__centroids[eid])) \
                                          for eid in self.__centroids], key=lambda t: t[1])

                bestmukey = max_similarity[0]

                # 只更新cluster_vec以及cluster_id
                if max_similarity[1] < self.__merge_sim_thres:
                    # eid_new = self.__date + "E" + str(self.__event_count)    # 20170620170000E0
                    # 記住此行, 修改merge時的id
                    self.__clusters_vec[event_id] = np.array(cluster_vec)
                    self.__clusters_id[event_id] = cluster_id
                    self.__centroids[event_id] = np.mean(self.__clusters_vec[event_id], axis=0)
                else:
                    self.__clusters_vec[bestmukey] = np.vstack((self.__clusters_vec[bestmukey], cluster_vec))
                    self.__clusters_id[bestmukey].extend(cluster_id)

                    # 之前曾經分裂過的event, 再次合併時必須將層次關係移除
                    if event_id in self.__son2father_event:
                        # self.__son2father_event[bestmukey] = self.__son2father_event[event_id]
                        father_event_id = self.__son2father_event[event_id]
                        if father_event_id in self.__father2son_event:
                            self.__father2son_event[father_event_id].remove(event_id)
                        self.__son2father_event.pop(event_id)

                    if event_id in self.__father2son_event:
                        self.__father2son_event.pop(event_id)

                pbar.update(1)
            pbar.close()
            time.sleep(0.3)

    # input = (cluster_id, [vecs]) // cluster info
    def split_cluster(self, cluster):
        """
        將評估過需要分裂的聚類放入function, 重新用online clustering聚類
        保留重新聚類的cluster[0]作為原本放入的聚類代表, 並在聚類時賦予父子關係
        :param cluster: (cluster_id:str, [vecs: numpy array])
        :return: cluster[1:] (排除掉cluster[0]的聚類)
        """
        event_id = cluster[0]
        event_vecs = cluster[1]
        clusters_id = self.__clusters_id[event_id]

        vectors = [ (clusters_id[i[0]], i[1]) for i in enumerate(event_vecs) ]
        n_clusters_vec, n_clusters_id, n_centroids = self.online_clustering(vectors=vectors, sim_thres=self.__subevent_sim_thres, mode='split', father_event_id=event_id)

        # replace the original cluster info with 1st newly generated cluster info
        self.__clusters_vec[event_id] = n_clusters_vec[event_id]
        self.__clusters_id[event_id] = n_clusters_id[event_id]
        self.__centroids[event_id] = n_centroids[event_id]

        outbase = self.outbase
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        outbase = os.path.join(outbase, "SplitResult")
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        outbase = os.path.join(outbase, "Split")
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        out_file = os.path.join(outbase, "split_event" + str(event_id))
        out = open(out_file, "w")
        out.write("Event " + str(event_id) + "\n")
        for key in n_clusters_id:
            news_id_all = n_clusters_id[key]
            out.write("Cluster " + str(key) + " num = " + str(len(news_id_all)) + "\n")
            for news_id in news_id_all:
                if news_id in self.__news:
                    result = self.__news[news_id]
                    out.write("Title: " + result['title'] + " Time: " + result['crawlTime'] + " Content: " + result['content'] + "\n")
        out.close()

        n_clusters_vec.pop(event_id)
        n_clusters_id.pop(event_id)
        n_centroids.pop(event_id)

        return (n_clusters_vec, n_clusters_id, n_centroids)

    def reevalute_centroids(self):
        """
        對cluster centroids重新評估, 重新計算一次聚類中心並評估是否需要分裂
        目前方法: MSE
        :return:
        """
        self.__centroids = { key : np.zeros(self.__dim) for key in self.__clusters_vec }
        clusters_vec = {}
        clusters_id = {}
        centroids = {}

        time.sleep(0.3)
        pbar = tqdm(total=len(self.__centroids), mininterval=1)
        for event_id in self.__clusters_vec:
            vecs = self.__clusters_vec[event_id]
            self.__centroids[event_id] += np.mean(vecs, axis=0)
            if len(vecs) > 1:
                mse = get_mse(vecs)
                self.mse.append(mse)
                if mse > self.__mse_thres:
                    cluster = (event_id, vecs)
                    n_clusters_vec, n_clusters_id, n_centroids = self.split_cluster(cluster=cluster)
                    clusters_vec.update(n_clusters_vec)
                    clusters_id.update(n_clusters_id)
                    centroids.update(n_centroids)
            pbar.update(1)

        pbar.close()
        time.sleep(0.3)
        return clusters_vec, clusters_id, centroids

    def rearrange_cluster(self):
        """
        對cluster做一次sort, 把較多數量的cluster放在上面以便比較
        :return:
        """
        event_id = sorted(self.__clusters_id.iteritems(), key=lambda v: len(v[1]), reverse=True)

        clusters_id = []
        for i in event_id:
            sort_id = i[0]
            clusters_id.append(i)

        return clusters_id

    def clustering_news(self, news_list):
        print "Vectorize"
        vectors = self.vectorize_mongolist(news_list=news_list)

        print "Clustering"
        clusters_vec, clusters_id, centroids = self.online_clustering(vectors=vectors, sim_thres=self.__sim_thres, mode='clustering')
        print "cluster = ", len(clusters_id)

        return (clusters_vec, clusters_id, centroids)

    def merge_events(self, time_info, cluster_triple):
        (start_time_ts, start_time_t), _ = time_info
        print "Read events"
        self.read_events(t=start_time_t)

        print "Merge"
        print "previous cluster = ", len(self.__clusters_id)
        self.online_clustering_merge(cluster_triple=cluster_triple)
        print "merged cluster = ", len(self.__clusters_id)

    def reevaluate(self, time_info):
        print "Re-evaluate centroids"
        cluster_triple = self.reevalute_centroids()

        print "Merge split event"
        self.online_clustering_merge(cluster_triple=cluster_triple)

    def write_event(self, t):
        """
        創造新的event並寫入mongoDB
        :param t: end_time_t
        :return:
        """
        time.sleep(0.3)
        pbar = tqdm(total=len(self.__clusters_id), mininterval=1)
        events = []
        for event_id in self.__clusters_id:
            event_result = self.__event_reader.query_one_by_item({'_id': event_id})
            # 先尋找event collection是否包含event_id的事件
            # 沒有找到
            if not event_result:
                event_json = get_event_json()
                event_json['created'] = t
                event_json['updated'] = t
            # 找到
            else:
                event_json = event_result
                event_json['updated'] = t

            # 寫入 event 基本info
            event_json['_id'] = event_id
            article_count = event_json['count']
            articles = []
            for news_id in self.__clusters_id[event_id]:
                # 尋找news collection是否包含news_id的新聞
                if news_id in self.__news:
                    n_news_dict = {"id":"", "title":"", "url":"", "publishTime":"", "abstract":""}
                    news_dict = self.__news[news_id]
                    n_news_dict['id'] = news_dict['_id']
                    n_news_dict['title'] = news_dict['title']
                    n_news_dict['url'] = news_dict['url']
                    n_news_dict['publishTime'] = news_dict['publishTime']
                    n_news_dict['abstract'] = news_dict['title']
                    articles.append(n_news_dict)
                    article_count += 1
            # articles
            event_json['count'] = article_count
            event_json['articles'] = articles

            # 寫入 event 的父子關係
            if event_id in self.__son2father_event:
                father_event_id = self.__son2father_event[event_id]
                event_json['father'] = father_event_id

            if event_id in self.__father2son_event:
                son_event_set = self.__father2son_event[event_id]
                event_json['childrens'] = list(son_event_set)

            # 寫入event的關鍵要素
            # keynews
            event_vecs = self.__clusters_vec[event_id]
            centroid_vec = self.__centroids[event_id]
            dist_list = [ (vid, np.sqrt(np.sum(np.square(vec - centroid_vec))) ) for vid, vec in enumerate(event_vecs) ]
            sort_dist = max(dist_list, key=lambda v:v[1])
            key_news_id = self.__clusters_id[event_id][sort_dist[0]]
            news_dict = self.__news[key_news_id]
            key_news_dict = {"id": "", "title": "", "url": "", "publishTime": "", "abstract": ""}
            key_news_dict['id'] = news_dict['_id']
            key_news_dict['title'] = news_dict['title']
            key_news_dict['url'] = news_dict['url']
            key_news_dict['publishTime'] = news_dict['publishTime']
            key_news_dict['abstract'] = get_content_abs(dim=self.__dim, content=news_dict['content'], centroid=centroid_vec, r=0.2)
            event_json['keynews'] = key_news_dict

            # 寫入event的相似事件, 僅僅會link上本次生成或讀取的event, 存在於collection內已經過期的event不影響
            centroid_vec = self.__centroids[event_id]
            score_list = [ (event_id, eid, cal_similarity(centroid_vec, vec)) for eid, vec in self.__centroids.iteritems() ]
            sort_scores = sorted(score_list, key=lambda v:v[-1], reverse=True)[1:]

            # print sort_scores
            related_events = []
            for score in sort_scores:
                if score[-1] > 0.4:
                    # r_event = {'id':"", 'label':"", score:0}
                    r_event = {}
                    rid = score[1]
                    # rlabel = self.__news[rid]['label']
                    r_event['id'] = rid
                    r_event['score'] = score[-1]
                    related_events.append(r_event)

            event_json['relatedEvents'] = related_events

            events.append(event_json)
            # self.__event_reader.save_item(event_json)
            pbar.update(1)
        pbar.close()
        time.sleep(0.3)

        for event in events:
            self.__event_reader.save_item(event)

    def write_result(self):
        """
        輸出聚類結果
        :return:
        """
        outbase = self.outbase
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        out = open(os.path.join(outbase, str(self.__date)), "w")
        # id_out = open(os.path.join(outbase, "cid"), "w")
        clusters_id = self.rearrange_cluster()

        for event_id, cluster in clusters_id:
            out.write("Cluster " + str(event_id) + " num = " + str(len(cluster)) + "\n")
            # id_out.write(json.dumps(cluster) + "\n")
            if len(cluster) == 1:
                self.__single_count += 1
            for news_id in cluster:
                if news_id in self.__news:
                    result = self.__news[news_id]
                    out.write("Title: " + result['title'] + " Time: " + result['crawlTime'] + " Content: " + result['content'] + "\n")

        out.close()
        # id_out.close()

        mse_out = open(os.path.join(outbase, "mse"+str(self.__date)), "w")
        for mse in sorted(self.mse, reverse=True):
            mse_out.write(str(mse) + "\n")
        mse_out.close()

    def write_log(self, time_info):
        """
        輸出本次聚類的相關log,
        :param time_info:
        :return:
        """
        (start_time_ts, start_time_t), (end_time_ts, end_time_t) = time_info
        outbase = self.outbase
        if not os.path.exists(outbase):
            os.mkdir(outbase)

        paras = open(os.path.join(outbase, "log_"+str(self.__date)+".txt"), "w")
        paras.write("Cost time = " + str(time.time() - self.__start) + "\n")
        paras.write("Start time: " + start_time_t + "\n")
        paras.write("End time: " + end_time_t + "\n")
        paras.write("Similarity = " + str(self.__sim_thres) + "\n")
        paras.write("Merge similarity = " + str(self.__merge_sim_thres) + "\n")
        paras.write("Subevent similarity = " + str(self.__subevent_sim_thres) + "\n")
        paras.write("Mse thres = " + str(self.__mse_thres) + "\n")
        paras.write("Total news num = " + str(self.__news_count) + "\n")
        paras.write("Single event = " + str(self.__single_count) + "\n")
        paras.write("Events = " + str(len(self.__clusters_id)) + "\n")
        paras.close()

    def output(self, time_info):
        _, (end_time_ts, end_time_t) = time_info
        print "Write event"
        self.write_event(t=end_time_t)
        self.write_result()
        self.write_log(time_info=time_info)

    def main(self, news_list, time_info):
        """

        :param news_list: 讀入時間段內全部新聞
        :param time_info: (start_time, end_time) 的封裝
        :param start_time: 開始時間 (start_time_ts, start_time_t) : (float, string)
        :param end_time: 結束時間 (end_time_ts, end_time_t) : (float, string)
        :return:
        """
        (start_time_ts, start_time_t), (end_time_ts, end_time_t) = time_info
        self.__date = self.__event_reader.create_event_id(t=start_time_t)
        cluster_triple = self.clustering_news(news_list=news_list)
        self.merge_events(time_info=time_info, cluster_triple=cluster_triple)
        self.reevaluate(time_info=time_info)
        self.output(time_info=time_info)
        print "-----------------------"