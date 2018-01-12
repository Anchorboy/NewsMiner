# -*- coding:utf-8 -*-
import os
import json
import numpy as np
from tqdm import tqdm
from datetime import *
import time
from header import get_event_json
from utils.function import Function
from utils.reader import EventReader, NewsReader


class Model():
    def __init__(self, dim, class_file, news_reader, event_reader, sim_thres=0.7, merge_sim_thres=0.7, subevent_sim_thres=0.7):
        self._func = Function()
        self._func.load_word_model(dim=dim, class_file=class_file)
        self.__dim = dim
        self.__sim_thres = sim_thres
        self.__merge_sim_thres = merge_sim_thres
        self.__subevent_sim_thres = subevent_sim_thres
        self.__mse_thres = 5e-6
        self.__cos_thres = 0.2
        self.__news = {}
        self.__events = {}
        self.__updated_events = {}
        self.__clusters_vec = {}
        self.__clusters_id = {}
        self.__centroids = {}
        self.__son2father_event = {} # single id: str
        self.__father2son_event = {} # son set: set of str
        self.mse = []
        self.cos = []
        self.cos_std = []
        self.__min_news_len = 80
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
        self.logbase = os.path.join(current_base, "log")

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
                vector = self._func.vectorize_single_news(dim=self.__dim, news_dict=news_dict)
                # vector = vectorize_with_dis(dim=self.__dim, news_dict=news_dict)
                vectors.append((news_id, vector))
            pb += 1
            pbar.update(1)

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
                max_similarity = max([(key, self._func.cal_similarity(vec, centroids[key])) \
                                      for key in centroids], key=lambda t: t[1])
            except:
                max_similarity = (0, 0)

            bestmukey = max_similarity[0]

            # 最大相似度小於相似度閾值, 產生新事件
            if max_similarity[1] < sim_thres:
                # 父事件, split分裂時紀錄father event
                if has_father_event:
                    key = father_event_id
                    has_father_event = False
                # 新事件
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

    def read_events(self, start_time_t):
        """
        從mongoDB event collection 讀取上一個階段聚類完成的event, 讀取後將event放入 self.__events 存儲
        在當中也會讀取mongoDB news collection, 並對於讀取的news作文檔向量化
        :param t: time string
        :return: event count: int
        """
        result = self.__event_reader.query_recent_events_by_time(t=start_time_t)

        time.sleep(0.3)
        pbar = tqdm(total=result.count(), mininterval=0.5)
        event_count = 0
        for event in result:
            event_id = event['_id']
            self.__events[event_id] = event
            self.__updated_events[event_id] = False

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
                        news_vec_in_event.append(self._func.vectorize_single_news(dim=self.__dim, news_dict=result))
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
            self.__clusters_vec[event_id] = np.asarray(news_vec_in_event, dtype=np.float)
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

                max_similarity = max([(eid, self._func.cal_similarity(centroid, self.__centroids[eid])) \
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
                    # merge到現有的event中, 並紀錄是否該event有更新的news, 如果有則為true
                    self.__updated_events[bestmukey] = True

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
        # self.__centroids = { key : np.mean(self.__clusters_vec[key], axis=0) for key in self.__clusters_vec }
        clusters_vec = {}
        clusters_id = {}
        centroids = {}

        time.sleep(0.3)
        pbar = tqdm(total=len(self.__centroids), mininterval=1)
        for event_id in self.__clusters_vec:
            vecs = self.__clusters_vec[event_id]
            self.__centroids[event_id] = np.mean(vecs, axis=0)
            cent_vec = self.__centroids[event_id]
            if len(vecs) > 1:
                # mse = self._func.get_mse(vecs, cent_vec)
                cos, cos_std = self._func.get_cos(vecs, cent_vec)
                # self.mse.append(mse)
                self.cos.append(cos)
                self.cos_std.append(cos_std)
                # if cos > 0.2:
                if cos_std > 0.055:
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
        self.read_events(start_time_t=start_time_t)

        print "Merge"
        print "previous cluster = ", len(self.__clusters_id)
        self.online_clustering_merge(cluster_triple=cluster_triple)
        print "merged cluster = ", len(self.__clusters_id)

    def reevaluate(self, time_info):
        print "Re-evaluate centroids"
        cluster_triple = self.reevalute_centroids()

        print "Merge split event"
        self.online_clustering_merge(cluster_triple=cluster_triple)

    def write_event(self, start_time_t):
        """
        創造新的event並寫入mongoDB, event格式包含在header.py中
        :param t: end_time_t 改為 start_time_t
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
                event_json['created'] = start_time_t
                event_json['updated'] = start_time_t
            # 找到
            else:
                event_json = event_result
                # 由於不是每個讀取的event都有更新, 因此僅僅紀錄更新過的event並將其寫回mongodb, 如果沒有更新就直接跳過
                if not self.__updated_events[event_id]:
                    pbar.update(1)
                    continue
                event_json['updated'] = start_time_t

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
                    n_news_dict['abstract'] = self._func.simple_content_abs(news_dict['content'])
                    articles.append(n_news_dict)
                    article_count += 1
            # articles
            event_json['count'] = article_count
            event_json['articles'] = articles

            # 寫入 event 的父子關係
            if event_id in self.__son2father_event:
                father_event_id = self.__son2father_event[event_id]
                event_json['father'] = father_event_id
            # 在這裡插入如果有子事件分裂的話, 則對該事件停止繼續合併的策略
            if event_id in self.__father2son_event:
                son_event_set = self.__father2son_event[event_id]
                event_json['childrens'] = list(son_event_set)
                event_json['closed'] = start_time_t

            # 寫入event的關鍵要素
            # keynews
            event_vecs = self.__clusters_vec[event_id]
            centroid_vec = self.__centroids[event_id]
            sim_list = [ (vid, self._func.cal_similarity(vec, centroid_vec) ) for vid, vec in enumerate(event_vecs) ]
            sort_dist = max(sim_list, key=lambda v:v[1])
            key_news_id = self.__clusters_id[event_id][sort_dist[0]]
            news_dict = self.__news[key_news_id]
            key_news_dict = {"id": "", "title": "", "url": "", "publishTime": "", "abstract": ""}
            key_news_dict['id'] = news_dict['_id']
            key_news_dict['title'] = news_dict['title']
            key_news_dict['url'] = news_dict['url']
            key_news_dict['publishTime'] = news_dict['publishTime']
            key_news_dict['abstract'] = self._func.get_content_abs(dim=self.__dim, content=news_dict['content'], centroid=centroid_vec, r=0.1)
            event_json['keynews'] = key_news_dict

            # 寫入event的相似事件, 僅僅會link上本次生成或讀取的event, 存在於collection內已經過期的event不影響
            centroid_vec = self.__centroids[event_id]
            score_list = [ (event_id, eid, self._func.cal_similarity(centroid_vec, vec)) for eid, vec in self.__centroids.iteritems() ]
            sort_scores = sorted(score_list, key=lambda v:v[-1], reverse=True)[1:]

            # print sort_scores
            k_realted_events = 15
            related_events = []
            for score in sort_scores[:k_realted_events]:
                if score[-1] > 0.6:
                    # r_event = {'id':"", 'label':"", score:0}
                    r_event = {}
                    rid = score[1]
                    # rlabel = self.__events[rid]['label']
                    # r_event['label'] = rlabel
                    r_event['id'] = rid
                    r_event['score'] = score[-1]
                    related_events.append(r_event)

            event_json['relatedEvents'] = related_events

            # 寫入實體列表, 並嘗試加上一個衰退率讓遠離事件中心的實體權重下降
            # 衰退率 decay = 1.0 / n_news, 此假設為聚類最外圍的新聞權重最小, 如果有10筆新聞, 最後一個新聞權重為0.1
            # sim_list 為最靠近中心的新聞sort list
            weight = 1.0
            decay = 1.0 / len(sim_list)
            # {word:score}
            keywords = {}
            persons = {}
            locations = {}
            organizations = {}
            entities = {}
            when = {}
            who = {}
            where = {}
            for vid, _ in sim_list:
                news_id = self.__clusters_id[event_id][vid]
                news = self.__news[news_id]
                # 關鍵詞抽取
                news_keywords_list = news['keywords']
                for keyword in news_keywords_list:
                    wd = keyword['word']
                    score = keyword['score']
                    if wd not in keywords:
                        keywords[wd] = score * weight
                    else:
                        keywords[wd] += score * weight

                # when, where, who抽取
                news_when_list = news['when']
                for keyword in news_when_list:
                    wd = keyword['word']
                    score = keyword['score']
                    if wd not in when:
                        when[wd] = score * weight
                    else:
                        when[wd] += score * weight

                news_where_list = news['where']
                for keyword in news_where_list:
                    wd = keyword['word']
                    score = keyword['score']
                    if wd not in where:
                        where[wd] = score * weight
                    else:
                        where[wd] += score * weight

                news_who_list = news['who']
                for keyword in news_who_list:
                    wd = keyword['word']
                    score = keyword['score']
                    if wd not in who:
                        who[wd] = score * weight
                    else:
                        who[wd] += score * weight

                # person, locations, organizations抽取
                # 暫時先用count作為score, 之後考慮打分機制
                # 目前加上score考慮方式: 由count做打分, 並乘上weight, 隨著遠離cluster中心遞減
                news_persons_list = news['persons']
                for person in news_persons_list:
                    mention = person['mention']
                    count = person['count'] * weight
                    url = person['linkedURL']
                    if mention not in persons:
                        persons[mention] = {'count':count, 'linkedURL':url}
                    else:
                        persons[mention]['count'] += count

                news_locations_list = news['locations']
                for location in news_locations_list:
                    mention = location['mention']
                    count = location['count'] * weight
                    url = location['linkedURL']
                    if mention not in locations:
                        locations[mention] = {'count': count, 'linkedURL': url}
                    else:
                        locations[mention]['count'] += count

                news_organizations_list = news['organizations']
                for organization in news_organizations_list:
                    mention = organization['mention']
                    count = organization['count'] * weight
                    url = organization['linkedURL']
                    if mention not in organizations:
                        organizations[mention] = {'count': count, 'linkedURL': url}
                    else:
                        organizations[mention]['count'] += count
                weight -= decay
            # 把生成的全部要素關鍵放回event_json
            k_important = 10
            event_json['keywords'] = [{'word':word, 'score':'{:.2f}'.format(score)} for word, score in sorted(keywords.iteritems(), key=lambda x:x[1], reverse=True)[:k_important]]
            event_json['when'] = [{'word':word, 'score':'{:.2f}'.format(score)} for word, score in sorted(when.iteritems(), key=lambda x:x[1], reverse=True)[:k_important]]
            event_json['where'] = [{'word':word, 'score':'{:.2f}'.format(score)} for word, score in sorted(where.iteritems(), key=lambda x:x[1], reverse=True)[:k_important]]
            event_json['who'] = [{'word':word, 'score':'{:.2f}'.format(score)} for word, score in sorted(who.iteritems(), key=lambda x:x[1], reverse=True)[:k_important]]
            # count_list = []
            event_json['persons'] = [{'mention':mention, 'score':'{:.2f}'.format(info['count']), 'linkedURL':info['linkedURL']}
                                     for mention, info in sorted(persons.iteritems(), key=lambda x:x[1]['count'], reverse=True)[:k_important]]
            event_json['locationss'] = [{'mention':mention, 'score':'{:.2f}'.format(info['count']), 'linkedURL': info['linkedURL']}
                                     for mention, info in sorted(locations.iteritems(), key=lambda x:x[1]['count'], reverse=True)[:k_important]]
            event_json['organizations'] = [{'mention':mention, 'score':'{:.2f}'.format(info['count']), 'linkedURL': info['linkedURL']}
                                     for mention, info in sorted(organizations.iteritems(), key=lambda x:x[1]['count'], reverse=True)[:k_important]]
            # 先暫時以keywords最大的關鍵字作為label, 到時候可以換成mention或其他keywords
            event_json['label'] = " ".join(i['word'] for i in event_json['keywords'][:3])
            # 本方法為考量全部keywords > 0.6的關鍵字, 並串聯再一起
            # event_json['label'] = " ".join([keyword for keyword in event_json['keywords'] if keyword['score'] > 0.6])

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

        # mse_out = open(os.path.join(outbase, "mse"+str(self.__date)), "w")
        # for mse in sorted(self.mse, reverse=True):
        #     mse_out.write(str(mse) + "\n")
        # mse_out.close()

        cos_out = open(os.path.join(outbase, "cos" + str(self.__date)), "w")
        for cos in sorted(self.cos, reverse=True):
            cos_out.write(str(cos) + "\n")
        cos_out.close()

        cos_out = open(os.path.join(outbase, "cos_std" + str(self.__date)), "w")
        for cos in sorted(self.cos_std, reverse=True):
            cos_out.write(str(cos) + "\n")
        cos_out.close()

    def write_log(self, time_info):
        """
        輸出本次聚類的相關log,
        :param time_info:
        :return:
        """
        (start_time_ts, start_time_t), (end_time_ts, end_time_t) = time_info
        logbase = self.logbase
        if not os.path.exists(logbase):
            os.mkdir(logbase)

        params = {'cost':time.time()-self.__start,
                  'start':start_time_t,
                  'end':end_time_t,
                  'clustering_sim':self.__sim_thres,
                  'merge_sim':self.__merge_sim_thres,
                  'subevent_sim':self.__subevent_sim_thres,
                  # 'mse':self.__mse_thres,
                  'cos': self.__cos_thres,
                  'n_news':self.__news_count,
                  'n_single_event':self.__single_count,
                  'n_events':len(self.__clusters_id)}
        with open(os.path.join(logbase, "log_"+str(self.__date)+".json"), "w") as f:
            f.write(json.dumps(params))
        with open(os.path.join(logbase, "log.json"), "w") as f:
            f.write(json.dumps(params))

    def output(self, time_info):
        (start_time_ts, start_time_t), (end_time_ts, end_time_t) = time_info
        print "Write event"
        self.write_event(start_time_t=start_time_t)
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
        # 僅僅在有讀入新聞時才做clustering, 否則則直接留下log
        if news_list.count():
            cluster_triple = self.clustering_news(news_list=news_list)
            self.merge_events(time_info=time_info, cluster_triple=cluster_triple)
            self.reevaluate(time_info=time_info)
            self.output(time_info=time_info)
        else:
            self.write_log(time_info=time_info)
            print "no news in current time span"

def test_model():
    print "test starting"
    print "load function"
    f = Function()
    dim = 2200
    class_file = "utils/" + str(dim) + ".txt"
    sim = 0.7
    merge_sim = 0.7
    sub_sim = 0.7
    IP_PORT = "10.1.1.46:27017"
    print "---------------"

    print "test news/events reader"
    news_reader = NewsReader(uri=IP_PORT)
    event_reader = EventReader(uri=IP_PORT, event_name="test_en_event")
    # clear event collection
    event_reader.remove_collection()
    print "---------------"

    print "test clustering model"
    print "time 2016-07-25 16:00:00 to 2016-07-26 18:00:00"
    start_time_t = "2016-07-25 16:00:00"
    end_time_t = "2016-07-26 18:00:00"
    time_info = f.generate_timeinfo(start_time_t, end_time_t)

    print time_info
    news_list = news_reader.query_many_by_time(start_time=start_time_t, end_time=end_time_t)
    clustering = Model(sim_thres=sim,
                       merge_sim_thres=merge_sim,
                       subevent_sim_thres=sub_sim,
                       dim=dim,
                       class_file=class_file,
                       news_reader=news_reader,
                       event_reader=event_reader)
    clustering.main(news_list=news_list, time_info=time_info)

    print "time 2016-07-26 18:00:00 to 2016-07-27 18:00:00"
    start_time_t = "2016-07-26 18:00:00"
    end_time_t = "2016-07-27 18:00:00"
    time_info = f.generate_timeinfo(start_time_t, end_time_t)

    print time_info
    news_list = news_reader.query_many_by_time(start_time=start_time_t, end_time=end_time_t)
    clustering = Model(sim_thres=sim,
                       merge_sim_thres=merge_sim,
                       subevent_sim_thres=sub_sim,
                       dim=dim,
                       class_file=class_file,
                       news_reader=news_reader,
                       event_reader=event_reader)
    clustering.main(news_list=news_list, time_info=time_info)
    print "---------------"

if __name__ == "__main__":
    test_model()