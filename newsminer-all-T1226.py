# -*- coding: utf-8 -*-
#本程序主要功能是对利用word2vector得到各分词的词向量，对各个分词进行次了聚类，得到分词聚类的结果，用各分词的聚类结果用来
# 表示文档的向量，利用已表示的文档向量利用online的方法进行聚类的实验。
#本程序在版本1的基础上进行改进，本程序与1的不同是本程序先进行聚类再进行合并。
#本程序在0228的基础上进行了部分改进，之前的版本在原有的事件基础上进行更新后没有标志，不利于人工标注；此版本在此基础上进行
# 了改进，当本次聚类的事件与上一次有更新，设置一个更新的标签。
#在0718版本的基础上修改了，单篇新闻向量的存储方式，由于单篇新闻向量比较稀疏，造成存储空间极大，针对该问题对向量进行压缩，
#记录向量内的非零位置，只存储非零元素，在需要计算的时候再变换为2500维度的向量。这样会极大的节省存储空间，但是会影响运行效率。
# 必须做折中处理
#在1108版本上修改score计算，将score进行归一化
#1120版本在1116的基础上，针对实体中存在重复实体的现象进行了改进，消除了出现重复实体的情况
#在1128版本的基础上，修改事件ID的生成方法，消除了事件ID号重复的现象；修改原有label生成的方法，原来的方法是直接取关键新闻
    # 的标题，现在的方法是与关键词重复最多的标题作为事件的label，能在一定程度上解决事件label不准确的问题。
#在1219的基础上增加事件deleted字段，默认全部输出为False。

import codecs
import json
import re
import sys
from numpy import array
from numpy import dot
from numpy.linalg import norm
import os
import shutil
import time
import datetime
from pymongo import MongoClient
import pymongo
from math import log
from bson import ObjectId
import bson
import getopt
import threading
import redis
reload(sys)
sys.setdefaultencoding("utf-8")

#####配置文件的读取####
congFile=codecs.open("./confile.txt","r","utf-8")
cong_list= []
for fr in congFile:
    fr=fr.strip()
    name= fr.split(":",1)
    cong_list.append(name[1])
database_IP = str(cong_list[0])
dataport = int(cong_list[1])
dbname = str(cong_list[2])
news_connection= str(cong_list[3])
event_connection=str(cong_list[4])
clusterThreshold=float(cong_list[5])
mergeThreshold=float(cong_list[6])
varThreshold=float(cong_list[7])
dict_word_cluster_name=str(cong_list[8])
stopwordsfile=str(cong_list[9])
closedConnection=str(cong_list[10])
redisIP=str(cong_list[11])
redisport=int(cong_list[12])
redispassword=str(cong_list[13])
congFile.close()
closedID=[]
now=""
#################
def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False
def is_number(uchar):
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False
#读取mongo数据库中的新闻元素
def ReadmongoData(startTime,endTime):#连接mongo数据库，并按照指定的规则读取数据，存储为json格式数据
#def ReadmongoData(StartTime,EndTime)
    ####参数设置######
    #####时间参数格式%Y%m%d%H%M%S
    # start_time="2016-03-03 00:00:00"
    # end_time=  "2016-03-04 00:00:00"
    start_time=startTime
    end_time=endTime
    ########
    client = MongoClient(database_IP,dataport)
    #db_name = 'NES'  # 数据库名
    db_name = dbname
    db = client[db_name]
    connection = db[news_connection]  # 临时的新闻数据表，用于测试
    file_list=[]
####查询条件，根据情况适当增加
    for line in connection.find({"crawlTime": {"$gte": start_time, "$lt": end_time}}).sort([("crawlTime", pymongo.ASCENDING)]):
    #for line in connection.find({"crawlTime": {"$gt": start_time, "$lt": end_time}}):
        try:
            file_list.append(line)
        except Exception, e:
            print Exception, ":", e
    print "读取新闻数据完成！"
    return file_list
#对新闻进行聚类
def cluster(infile,threshold):
    print "开始聚类！"
    stopword=[]
    stopfile=codecs.open(stopwordsfile)
    for stopw in stopfile:
        stopword.append(stopw.strip())
    stopfile.close()
    thresholdvalue = threshold
    ###与mongo数据库直接连接后，数据的输入直接从数据库中读取，不需要读文件
    fileF =infile
    dict_word_cluster = json.load(open(dict_word_cluster_name))
    mountofcluster=2500#词聚类时是定的聚类数量
    count = 0  ####第几篇文档
    number_DocsinTopic = {}  # 用于记录每个topic下的文档数目，用于计算类的中心
    numoftopic = 0  #####topic的编号
    doc2vec = []
    topic2vec = {}
    DocumentinTopic = {}
    document = []
#######
    ######
    file_dict={}
    aa=0
    for lines in fileF:
        file_dict[aa]=lines
        aa +=1
######
    for key in file_dict:
        x=file_dict[key][u"seggedContent"][0]
        document.append(key)
        list_tmp=[0.0 for i in range(mountofcluster)]
        for words in x.split(" "):
            word=words.split("/")[0]
            if word in stopword:
                continue
            if is_alphabet(word):
                continue
            if is_number(word):
                continue
            if len(word)<2:
                continue
            try:
                list_tmp[int(dict_word_cluster[word])] +=1.0
            except KeyError:  ####考虑到新闻中的分词结果有部分的词没出现在模型中
                pass
            except Exception, e:
                print Exception, ":", e
        narry=array(list_tmp)
        ##########新增单篇新闻的向量####
        file_dict[key][u"newsVector"]=narry.tolist()
        ####一篇文档的vector计算完毕，开始计算与之前topic的余弦相似度进行合并
        dict_similarity = {}
        if count == 0:  ####第一篇文档，直接归为第一个topic
            #topic = "topic:0"
            topic = 0
            tmp = []
            tmp.append(count)
            DocumentinTopic[topic] = tmp  #####将该文档的的标号归到该topic下
            topic2vec[topic] = narry
            number_DocsinTopic[topic] = 1.0
        else:  ###需要与前面的topic进行余弦相似度的比较
            for i in topic2vec:  #####可以进行改进，此时对所有的topic进行遍历，效率比较低，可以增加实体的索引，提高速度
                dp = dot(narry, topic2vec[i])
                np = norm(narry) * norm(topic2vec[i])+0.01
                sim = dp / np
                dict_similarity[i] = sim
            #######找出最大的余弦相似度
            b = dict_similarity.values()
            sim = b[0]
            topic = dict_similarity.keys()[0]
            for key, value in dict_similarity.items():
                if value >= sim:
                    topic = key
                    sim = value
            #print sim
            if sim >thresholdvalue:  #########余弦相似度阈值
                ######找到同一个topic进行合并
                DocumentinTopic[topic].append(count)
                topic2vec[topic] = (topic2vec[topic] * number_DocsinTopic[topic] + narry) / (
                number_DocsinTopic[topic] + 1.0)
                number_DocsinTopic[topic] += 1.0
            else:  #####新的topic
                topic=len(topic2vec)
                tmp = []
                tmp.append(count)
                DocumentinTopic[topic] = tmp
                topic2vec[topic] = narry
                number_DocsinTopic[topic] = 1.0
        count += 1
    print "clustering finished!"
    ##########输出事件聚类中心向量
    for i in topic2vec:
        topic2vec[i] = topic2vec[i].tolist()
###@#############
    fina_list=[]
    for i in DocumentinTopic:
        topic_dict = {}
        topic_dict[u"id"] = i
        topic_dict[u"label"]=""
        topic_dict[u"deleted"]="false"
        topic_dict[u"created"]=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        topic_dict[u"updated"]=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        topic_dict[u"closed"]=False
        topic_dict[u"eventVector"] = topic2vec[i]
        topic_dict["persons"] = []
        topic_dict["organizations"] = []
        topic_dict["locations"] = []
        topic_dict[u"keywords"] = []
        topic_dict[u"articles"] = []
        topic_dict[u"when"] = []
        topic_dict[u"where"] = []
        topic_dict[u"who"] = []
        for j in DocumentinTopic[i]:
            singalArticle_dict = {}
#0622####################################
            singalArticle_dict[u"id"] = file_dict[document[j]][u"_id"]##########
            singalArticle_dict[u"title"] = file_dict[document[j]][u"title"]
            singalArticle_dict[u"url"]= file_dict[document[j]][u"url"]
##0719修改部分#####
            singalArticle_dict[u"newsVector"]=[]
            for item in xrange(len(file_dict[document[j]][u"newsVector"])):
                dict_tmp = {}
                if file_dict[document[j]][u"newsVector"][item] >0:
                    dict_tmp[str(item)]=file_dict[document[j]][u"newsVector"][item]
                    singalArticle_dict[u"newsVector"].append(dict_tmp)
                del dict_tmp
            #singalArticle_dict[u"newsVector"] = file_dict[document[j]][u"newsVector"]
####0719##########
            ###旧的新闻格式还未增加该字段
            singalArticle_dict[u"publishTime"]=file_dict[document[j]][u"publishTime"]
            singalArticle_dict[u"image"] = file_dict[document[j]][u"image"]
            singalArticle_dict[u"publisher"] = file_dict[document[j]][u"publisher"]
            singalArticle_dict[u"category"] = file_dict[document[j]][u"category"]
            try:
                singalArticle_dict[u"abstract"] = file_dict[document[j]][u"content"][0:99]
            except:
                singalArticle_dict[u"abstract"]=""
            topic_dict[u"articles"].append(singalArticle_dict)
            for per in file_dict[document[j]][u"persons"]:
                per[u"idf"]=1.0
                topic_dict["persons"].append(per)
            for loc in file_dict[document[j]][u"locations"]:
                loc[u"idf"]=1.0
                topic_dict["locations"].append(loc)
            for org in file_dict[document[j]][u"organizations"]:
                org[u"idf"]=1.0
                topic_dict["organizations"].append(org)
            for kw in file_dict[document[j]][u"keywords"]:
                topic_dict[u"keywords"].append(kw)
####旧的字段没有，需要增加的部分
            for when in file_dict[document[j]][u"when"]:
                topic_dict[u"when"].append(when)
            for where in file_dict[document[j]][u"where"]:
                topic_dict[u"where"].append(where)
            for who in file_dict[document[j]][u"who"]:
                topic_dict[u"who"].append(who)
####
        topic_dict[u"count"]=len(topic_dict[u"articles"])
        fina_list.append(topic_dict)
    return fina_list
#对相同的事件进行合并
def merge(NumOfTopic,threshold,newevent,oldevent):
    print "开始合并！"
    ######
    thresholdvalue = threshold#合并的阈值，得提前设定
    dict_word_cluster = json.load(open(dict_word_cluster_name))
    mountofcluster = 2500  # 词聚类时是定的聚类数量
    count = 0  ####第几篇文档
    number_DocsinTopic = {}  # 用于记录每个topic下的文档数目，用于计算类的中心
    numoftopic = 0  #####topic的编号
    doc2vec = []
    topic2vec = {}
    DocumentinTopic= {}
    document={}
    document1={}
    normlization={}
##################################
    for line in oldevent:
        topic2vec[line[u"id"]] = array(line[u"eventVector"])
        number_DocsinTopic[line[u"id"]] = line[u"count"]
        normlization[line[u"id"]]=False
        ###############33
    file_dict = {}
    aa = 0
###############
    for line in newevent:
        ####
        narry = array(line[u"eventVector"])
        key=line[u"id"]
        ####一篇文档的vector计算完毕，开始计算与之前topic的余弦相似度进行合并
        dict_similarity = {}
        for i in topic2vec:  #####可以进行改进，此时对所有的topic进行遍历，效率比较低，可以增加实体的索引，提高速度
            dp = dot(narry, topic2vec[i])
            np = norm(narry) * norm(topic2vec[i]) + 0.01
            sim = dp / np
            dict_similarity[i] = sim
            #######找出最大的余弦相似度
        b = dict_similarity.values()
        sim = b[0]
        topic = dict_similarity.keys()[0]
        for keys, value in dict_similarity.items():
            if value >= sim:
                topic = keys#enent的id号
                sim = value
        #print sim
        if sim > thresholdvalue:  #########余弦相似度阈值
            ######找到同一个topic进行合并
            for result in oldevent:
                if result[u"id"] == int(topic):
                    for event in newevent:
                        if event[u"id"]==int(key):
                            result[u"updated"]=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
#######################################1117修改
                            if normlization[result[u"id"]] ==False:
                                for item in result[u"keywords"]:
                                    item[u"score"] = item[u"score"] * result[u"count"]
                                for item in result[u"when"]:
                                    item[u"score"] = item[u"score"] * result[u"count"]
                                for item in result[u"where"]:
                                    item[u"score"] = item[u"score"] * result[u"count"]
                                for item in result[u"who"]:
                                    item[u"score"] = item[u"score"] * result[u"count"]
                                normlization[result[u"id"]]=True
############################################################
                            for article in event[u"articles"]:
                                singalArticle_dict = {}
                                singalArticle_dict[u"id"]=article[u"id"]
                                singalArticle_dict[u"title"] = article[u"title"]
                                singalArticle_dict[u"url"] = article[u"url"]
                                singalArticle_dict[u"publishTime"] = article[u"publishTime"]
                                singalArticle_dict[u"image"] = article[u"image"]
                                singalArticle_dict[u"publisher"] = article[u"publisher"]
                                singalArticle_dict[u"category"] = article[u"category"]
                                try:
                                    singalArticle_dict[u"abstract"] = article[u"abstract"]
                                except Exception, e:
                                    print Exception, ":", e
                                    singalArticle_dict[u"abstract"]=""
# ######0719修改单篇新闻向量的存储方式
#                                 singalArticle_dict[u"newsVector"] = []
#                                 for item in xrange(len(article[u"newsVector"])):
#                                     dict_tmp = {}
#                                     if article[u"newsVector"][item] > 0:
#                                         dict_tmp[item] = article[u"newsVector"][item]
#                                         singalArticle_dict[u"newsVector"].append(dict_tmp)
#                                     del dict_tmp
                                singalArticle_dict[u"newsVector"] = article[u"newsVector"]
# ########0719修改###############
                                #singalArticle_dict[u"time"] = event[u"time"]
                                result[u"articles"].append(singalArticle_dict)
                            result[u"count"] = len(result[u"articles"])
#############################333这部分要修改，结构与现在的结构不一致
                            for per in event[u"persons"]:
                                result["persons"].append(per)
                            for loc in event[u"locations"]:
                                result["locations"].append(loc)
                            for org in event[u"organizations"]:
                                result["organizations"].append(org)
                            for kw in event[u"keywords"]:
                                result[u"keywords"].append(kw)
                            for when in event[u"when"]:
                                result[u"when"].append(when)
                            for where in event[u"where"]:
                                result[u"where"].append(where)
                            for who in event[u"who"]:
                                result[u"who"].append(who)
                            topic2vec[topic] = (topic2vec[topic] * (result[u"count"]-len(
                                event[u"articles"])) + narry * len(event[u"articles"])) / (result[u"count"])
                            number_DocsinTopic[topic] += len(event[u"articles"])
                            break
                    break
            ####
        else:  #####新的topic
######################需要事先统计数据库中的事件总数，然后进行累加
            #topic = len(topic2vec)#需要修改。。。。。。。
            topic=NumOfTopic
            NumOfTopic +=1
            topic_dict = {}
            topic_dict[u"id"] = int(topic)
            topic_dict[u"label"]=""
            topic_dict[u"deleted"]="false"
            topic_dict[u"created"]=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
            topic_dict[u"updated"]=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
            topic_dict[u"closed"]=False
#################
            # topic_dict[u"update"]=False
#############
            topic_dict["persons"] = []
            topic_dict["organizations"] = []
            topic_dict["locations"] = []
            topic_dict[u"keywords"] = []
            topic_dict[u"articles"] = []
            topic_dict[u"where"] = []
            topic_dict[u"who"] = []
            topic_dict[u"when"] = []
            for event in newevent:
                if event[u"id"] == int(key):
                    for article in event[u"articles"]:
                        singalArticle_dict = {}
                        singalArticle_dict[u"id"] = article[u"id"]
                        singalArticle_dict[u"title"] = article[u"title"]
                        singalArticle_dict[u"url"] = article[u"url"]
                        singalArticle_dict[u"publishTime"] = article[u"publishTime"]
                        singalArticle_dict[u"image"] = article[u"image"]
                        singalArticle_dict[u"publisher"] = article[u"publisher"]
                        singalArticle_dict[u"category"] = article[u"category"]
                        try:
                            singalArticle_dict[u"abstract"] = article[u"abstract"]
                        except Exception, e:
                            print Exception, ":", e
                            singalArticle_dict[u"abstract"] =""
# ######0719修改单篇新闻向量的存储方式
#                         singalArticle_dict[u"newsVector"] = []
#                         for item in xrange(len(article[u"newsVector"])):
#                             dict_tmp = {}
#                             if article[u"newsVector"][item] > 0:
#                                 dict_tmp[item] = article[u"newsVector"][item]
#                                 singalArticle_dict[u"newsVector"].append(dict_tmp)
#                             del dict_tmp
                        #singalArticle_dict[u"newsVector"] = article[u"newsVector"]
########0719修改###############
                        singalArticle_dict[u"newsVector"] = article[u"newsVector"]
                        topic_dict[u"articles"].append(singalArticle_dict)
                    topic_dict[u"count"] = len(topic_dict[u"articles"])
                    for per in event[u"persons"]:
                        topic_dict["persons"].append(per)
                    for loc in event[u"locations"]:
                        topic_dict["locations"].append(loc)
                    for org in event[u"organizations"]:
                        topic_dict["organizations"].append(org)
                    for kw in event[u"keywords"]:
                        topic_dict[u"keywords"].append(kw)
                    for when in event[u"when"]:
                        topic_dict[u"when"].append(when)
                    for where in event[u"where"]:
                        topic_dict[u"where"].append(where)
                    for who in event[u"who"]:
                        topic_dict[u"who"].append(who)
                    break
            oldevent.append(topic_dict)
            ###############################
################
            normlization[int(topic)]=True
###############
            topic2vec[topic] = narry
            number_DocsinTopic[topic] = event[u"count"]
        count += 1
    print "merge finished!"
    ##########输出事件聚类中心向量
    for i in topic2vec:
        topic2vec[i] = topic2vec[i].tolist()
        for line in oldevent:
            if line[u"id"]==i:
                line[u"eventVector"]=topic2vec[i]#更新事件的向量，事件的id号重要
    return oldevent #返回最终的事件结果
#读取mongo数据库中事件的元素
def ReadMongoEventData(startTime,endTime):
    ####参数设置######
    #####时间参数格式%Y%m%d%H%M%S
    # start_time = "2017-06-30 00:00:00"
    # end_time = "2017-06-30 19:00:00"
    start_time =startTime
    end_time = endTime
    ########
    client = MongoClient(database_IP,dataport)
    db_name =dbname# 'NES'  # 数据库名
    db = client[db_name]
    connection=db[event_connection]#临时的新闻数据表，用于测试
    #connection.drop()
    ######
    file_list = []
    ####查询条件，根据情况适当增加
    ##返回topic的总数
    num = connection.count()
    # for line in connection.find({"updated": {"$gt": start_time, "$lt": end_time}}).sort(
    #         [("updated", pymongo.ASCENDING)]):
    for line in connection.find({"updated": {"$gte": start_time, "$lt": end_time},"closed":False,"count":{"$gte":3},"deleted":"false"}):
        try:
            file_list.append(line)
        except Exception, e:
            print Exception, ":", e
    return (file_list,num)
#将处理完成的事件信息存入事件库中
def WriteMongoEventData(EventFiles):
    outfile=codecs.open(r"./result.txt","w","utf-8")
    client = MongoClient(database_IP,dataport)
    db_name =dbname# 'NES'  # 数据库名
    db = client[db_name]
    connection=db[event_connection]#临时的新闻数据表，用于测试
    #connection.drop()
    pool = redis.ConnectionPool(host=redisIP, port=redisport, password=redispassword,db=1)
    r = redis.Redis(connection_pool=pool)
    for line in EventFiles:
        #line[u"_id"] = time.strftime("%Y-%m-%d %H:%M:%S", line[u"_id"].generation_time.timetuple())+ObjectId().str
        #line[u"_id"]=time.mktime(line[u"_id"].generation_time.timetuple())
        # for art in line[u"articles"]:
        #     del art[u"newsVector"]
        if line.has_key(u"_id")==False:
            line[u"_id"]=time.strftime("%Y%m%d%H%M%S", time.localtime())+str(ObjectId())

            if line[u"updated"] >= now:
                r.set(line["_id"], json.dumps(line,ensure_ascii=False))
            try:
                connection.insert(line)
            except Exception, e:
                print Exception, ":", e
                outfile.write(json.dumps(line, ensure_ascii=False) + "\n")
        else:
            if line[u"updated"] >= now:
                r.set(line["_id"], json.dumps(line,ensure_ascii=False))
            try:
                connection.update({"_id": line[u"_id"]}, {'$set': line})
            except Exception, e:
                print Exception, ":", e
                outfile.write(json.dumps(line, ensure_ascii=False) + "\n")
            #     print len(bson.dumps(line))
        #outfile.write(json.dumps(line, ensure_ascii=False) + "\n")
    outfile.close()
#计算事件内，新闻的相似度方差，并根据与聚类中心的聚类，对新闻进行排序
def variance(inputevent):
    similarity_dict = {}
    event_narry = array(inputevent[u"eventVector"])
    ################相似度列表初始化list
    sim_list = []
    ####################
    for article in inputevent[u"articles"]:
        news_vec=[0.0 for ii in xrange(2500)]
        for item in article[u"newsVector"]:
            news_vec[int(item.keys()[0])]=item.values()[0]
        dp = dot(event_narry, array(news_vec))
        np = norm(event_narry) * norm(array(news_vec)) + 0.0001
        # dp = dot(event_narry, array(article[u"newsVector"]))
        # np = norm(event_narry) * norm(array(article[u"newsVector"])) + 0.0001
        sim = dp / np
        similarity_dict[article[u"id"]] = sim
        sim_list.append(sim)
        #################计算相似度阈值的方差
    sim_array = array(sim_list)
    var = sim_array.var()
    return var
    ####################
#对事件元素信息进行排序
def sort(inputevent):  #输入需要排序的事件类
    ##计算事件内新闻与聚类中心的相似度
    Jaccard_dict={}
    keywords_list=inputevent[u"keywords"][:50]
    similarity_dict = {}
    event_narry = array(inputevent[u"eventVector"])
    ####################
    for article in inputevent[u"articles"]:
        Jaccard_dict[article[u"id"]] = 0.0
        for kw in keywords_list:
            try:
                if kw[u"word"] in article[u"title"]:
                    Jaccard_dict[article[u"id"]] += 1.0
            except Exception ,e:
                pass
        news_vec=[0.0 for ii in xrange(2500)]
        for item in article[u"newsVector"]:
            news_vec[int(item.keys()[0])]=item.values()[0]
        dp = dot(event_narry, array(news_vec))
        np = norm(event_narry) * norm(array(news_vec)) + 0.0001
        # dp = dot(event_narry, array(article[u"newsVector"]))
        # np = norm(event_narry) * norm(array(article[u"newsVector"])) + 0.0001
        sim = dp / np
        similarity_dict[article[u"id"]] = sim
    ####进行排序
    SortJaccard = sorted(Jaccard_dict.items(), key=lambda d: d[1], reverse=True)
    SortResult = sorted(similarity_dict.items(), key=lambda d: d[1], reverse=True)
    flag = True
    article_list=[]
    for id, sim in SortJaccard:
        for article in inputevent[u"articles"]:
            if article[u"id"] == id:
                inputevent[u"label"] = article[u"title"]
                break
        break
    for id, sim in SortResult:
        for article in inputevent[u"articles"]:
            if article[u"id"] == id:
                article[u"score"] = sim
                break
        article_list.append(article)
        if flag ==True:
            flag =False
            inputevent[u"keynews"]=article
            #inputevent[u"label"] = article[u"title"]
            try:
                inputevent[u"category"]=article[u"category"]
            except  Exception, e:
                pass
    inputevent[u"articles"]=article_list
    ####对事件的5w以及关键词进行降序排列,存在关键词以及事件内的元素重复，需要去重
    if inputevent[u"count"]>1:
        squence =0
        while squence<(len(inputevent[u"keywords"])-1):
            tmp_list=[]
            tmp_list=inputevent[u"keywords"][(squence+1):]
            item =0
            count_tmp=0
            while item <len(tmp_list):
            #for item in xrange(len(tmp_list)):
                if inputevent[u"keywords"][squence][u"word"]==tmp_list[item][u"word"]:
                    inputevent[u"keywords"][squence][u"score"] +=tmp_list[item][u"score"]
                    #del tmp_list[item]
                    del inputevent[u"keywords"][item+squence+1-count_tmp]
                    count_tmp +=1
                item +=1
            squence +=1
        squence =0
        while squence < (len(inputevent[u"where"])-1):
            tmp_list=[]
            tmp_list=inputevent[u"where"][(squence+1):]
            item =0
            count_tmp=0
            while item <len(tmp_list):
            #for item in xrange(len(tmp_list)):
                if inputevent[u"where"][squence][u"word"]==tmp_list[item][u"word"]:
                    inputevent[u"where"][squence][u"score"] +=tmp_list[item][u"score"]
                    #del tmp_list[item]
                    del inputevent[u"where"][item+squence+1-count_tmp]
                    count_tmp +=1
                item +=1
            squence +=1
        squence =0
        while squence < (len(inputevent[u"when"])-1):
            tmp_list=[]
            tmp_list=inputevent[u"when"][(squence+1):]
            item =0
            count_tmp=0
            while item <len(tmp_list):
            #for item in xrange(len(tmp_list)):
                if inputevent[u"when"][squence][u"word"]==tmp_list[item][u"word"]:
                    inputevent[u"when"][squence][u"score"] +=tmp_list[item][u"score"]
                    #del tmp_list[item]
                    del inputevent[u"when"][item+squence+1-count_tmp]
                    count_tmp +=1
                item +=1
            squence +=1
        squence =0
        while squence < (len(inputevent[u"who"])-1):
            tmp_list=[]
            tmp_list=inputevent[u"who"][(squence+1):]
            item =0
            count_tmp=0
            while item <len(tmp_list):
            #for item in xrange(len(tmp_list)):
                if inputevent[u"who"][squence][u"word"]==tmp_list[item][u"word"]:
                    inputevent[u"who"][squence][u"score"] +=tmp_list[item][u"score"]
                    #del tmp_list[item]
                    del inputevent[u"who"][item+squence+1-count_tmp]
                    count_tmp +=1
                item +=1
            squence +=1
    inputevent[u"keywords"] = sorted(inputevent["keywords"], key=lambda d: d["score"], reverse=True)
    inputevent[u"where"] = sorted(inputevent["where"], key=lambda d: d["score"], reverse=True)
    inputevent[u"when"] = sorted(inputevent["when"], key=lambda d: d["score"], reverse=True)
    inputevent[u"who"] = sorted(inputevent["who"], key=lambda d: d["score"], reverse=True)

    #keywordsScore = inputevent[u"keywords"][0][u"score"]
    # 进行归一化
    for keyword in inputevent[u"keywords"]:
        keyword[u"score"] = float(keyword[u"score"]) / inputevent[u"count"]
    # whereScore = inputevent[u"where"][0][u"score"]
    # # 进行归一化
    for where in inputevent[u"where"]:
        where[u"score"] = float(where[u"score"]) / inputevent[u"count"]
    # whenScore = inputevent[u"when"][0][u"score"]
    # # 进行归一化
    for when in inputevent[u"when"]:
        when[u"score"] = float(when[u"score"]) / inputevent[u"count"]
    # whoScore = inputevent[u"who"][0][u"score"]
    # # 进行归一化
    for who in inputevent[u"who"]:
        who[u"score"] = float(who[u"score"]) / inputevent[u"count"]
    #########进行实体关系排序
    #组织中存在大量的新闻机构应该定义策略过滤。
    orgStopwods=[u"环球",u"腾讯",u"中新网",u"新华社",u"路透社",u"新浪",u"新浪网",u"网易",u"新华网",u"搜狐",u"中文",u"英文",u"汉语",u"新华",u"话语",u"太极",u"新浪网",u"八大",u"凡是",
                 u"格格",u"路透",u"新民",u"西游",u"西游记",u"金鸡",u"文华",u"顺顺",u"同心",u"周易",u"樱花",u"法语",u"法新社",u"可乐",u"汉族",u"华表奖",u"贝贝",u"朝外",u"朝鲜战争",u"波波",u"裕华",u"华以",u"鹏程",
                 u"安乐",u"英语",u"富豪",u"中华",u"论语",u"抗日战争",u"藏语",u"藏族",u"海选",u"布依族"]
    if inputevent[u"count"] > 1:
        # for squence in xrange(len(inputevent[u"persons"])):
        #     inputevent[u"persons"][squence][u"idf"]=1.0
        squence=0
        while squence<(len(inputevent[u"persons"])-1):
        #for squence in xrange(len(inputevent[u"persons"])):
            tmp_list=[]
            tmp_list=inputevent[u"persons"][(squence+1):]
            item = 0
            count_tmp =0
            while item < len(tmp_list):
            #for item in xrange(len(tmp_list)):
                if inputevent[u"persons"][squence][u"mention"]==tmp_list[item][u"mention"]:
                    inputevent[u"persons"][squence][u"count"] +=tmp_list[item][u"count"]
                    inputevent[u"persons"][squence][u"idf"] +=1.0
                    #del tmp_list[item]
                    del inputevent[u"persons"][item+squence+1-count_tmp]
                    count_tmp +=1
                item +=1
            squence +=1
        for squence in xrange(len(inputevent[u"persons"])):
            inputevent[u"persons"][squence][u"score"]=inputevent[u"persons"][squence][u"count"]/(log(inputevent[u"count"]/inputevent[u"persons"][squence][u"idf"],10)+0.01)
            # del inputevent[u"locations"][squence][u"idf"]
            # del inputevent[u"locations"][squence][u"count"]


        # for squence in xrange(len(inputevent[u"locations"])):
        #     inputevent[u"locations"][squence][u"idf"]=1.0
        squence=0
        while squence<(len(inputevent[u"locations"])-1):
        #for squence in xrange(len(inputevent[u"locations"])):
            tmp_list=[]
            tmp_list=inputevent[u"locations"][(squence+1):]
            item = 0
            count_tmp =0
            while item < len(tmp_list):
            #for item in xrange(len(tmp_list)):
                if inputevent[u"locations"][squence][u"mention"]==tmp_list[item][u"mention"]:
                    inputevent[u"locations"][squence][u"count"] +=tmp_list[item][u"count"]
                    inputevent[u"locations"][squence][u"idf"] +=1.0
                    #del tmp_list[item]
                    del inputevent[u"locations"][item+squence+1-count_tmp]
                    count_tmp +=1
                item +=1
            squence +=1
        for squence in xrange(len(inputevent[u"locations"])):
            inputevent[u"locations"][squence][u"score"]=inputevent[u"locations"][squence][u"count"]/(log(inputevent[u"count"]/inputevent[u"locations"][squence][u"idf"],10)+0.01)
            # del inputevent[u"locations"][squence][u"idf"]
            # del inputevent[u"locations"][squence][u"count"]

        # for squence in xrange(len(inputevent[u"organizations"])):
        #     inputevent[u"organizations"][squence][u"idf"]=1.0
        squence = 0
        while squence < (len(inputevent[u"organizations"]) - 1):
        #for squence in xrange(len(inputevent[u"organizations"])):
            tmp_list=[]
            tmp_list=inputevent[u"organizations"][(squence+1):]
            item = 0
            count_tmp =0
            while item < len(tmp_list):
            #for item in xrange(len(tmp_list)):
                if inputevent[u"organizations"][squence][u"mention"]==tmp_list[item][u"mention"]:
                    inputevent[u"organizations"][squence][u"count"] +=tmp_list[item][u"count"]
                    inputevent[u"organizations"][squence][u"idf"] +=1.0
                    #del tmp_list[item]
                    del inputevent[u"organizations"][item+squence+1-count_tmp]
                    count_tmp +=1
                item +=1
            squence +=1
        for squence in xrange(len(inputevent[u"organizations"])):
            inputevent[u"organizations"][squence][u"score"]=inputevent[u"organizations"][squence][u"count"]/(log(inputevent[u"count"]/inputevent[u"organizations"][squence][u"idf"],10)+0.01)
            if inputevent[u"organizations"][squence][u"mention"] in orgStopwods:
                inputevent[u"organizations"][squence][u"score"]=0.0
            # del inputevent[u"organizations"][squence][u"idf"]
            # del inputevent[u"organizations"][squence][u"count"]
    else:
        for squence in xrange(len(inputevent[u"persons"])):
            inputevent[u"persons"][squence][u"score"]=float(inputevent[u"persons"][squence][u"count"])
            #del inputevent[u"persons"][squence][u"count"]
        for squence in xrange(len(inputevent[u"locations"])):
            inputevent[u"locations"][squence][u"score"]=float(inputevent[u"locations"][squence][u"count"])
            #del inputevent[u"locations"][squence][u"count"]
        for squence in xrange(len(inputevent[u"organizations"])):
            inputevent[u"organizations"][squence][u"score"]=float(inputevent[u"organizations"][squence][u"count"])
            if inputevent[u"organizations"][squence][u"mention"] in orgStopwods:
                inputevent[u"organizations"][squence][u"score"]=0.0
            #del inputevent[u"organizations"][squence][u"count"]
    inputevent[u"persons"] = sorted(inputevent["persons"], key=lambda d: d["score"], reverse=True)
    inputevent[u"locations"] = sorted(inputevent["locations"], key=lambda d: d["score"], reverse=True)
    inputevent[u"organizations"] = sorted(inputevent["organizations"], key=lambda d: d["score"], reverse=True)
    #########
    if len(inputevent[u"persons"])>0:
        personScore = inputevent[u"persons"][0][u"score"]+0.00001
        # 进行归一化
        for person in inputevent[u"persons"]:
            person[u"score"] = person[u"score"] / personScore
    if len(inputevent[u"locations"])>0:
        locationScore = inputevent[u"locations"][0][u"score"]+0.00001
    #    进行归一化
        for location in inputevent[u"locations"]:
            location[u"score"] = location[u"score"] / locationScore
    if len(inputevent[u"organizations"])>0:
        organizationScore = inputevent[u"organizations"][0][u"score"]+0.00001
    # 进行归一化
        for organization in inputevent[u"organizations"]:
            organization[u"score"] = organization[u"score"] / organizationScore
    return inputevent  ###返回进行排序后的事件信息以及关键代表新闻
if __name__ == "__main__":
#def myProject(news_time1,news_time2,event_time1,event_time2):

    pool = redis.ConnectionPool(host=redisIP, port=redisport, password=redispassword)
    r = redis.Redis(connection_pool=pool)
    time1=time.time()
####时间窗口的划分，分为新闻时间窗口和事件时间窗口
    timeFile=codecs.open("./time.txt","r","utf-8")
    time_list=[]
    flag=True
    for fr in timeFile:
        fr=fr.strip()
        time_list.append(fr)
    timeFile.close()
    if time_list[len(time_list)-1]==time_list[len(time_list)-4]:
        starttime= time_list[0]
        endtime=time_list[1]
        flag=False
    else:
        for i in range(0,len(time_list)-4,1):
            if time_list[len(time_list)-1]==time_list[i]:
                endtime=time_list[i+1]
                starttime=time_list[i]
                flag=False
                break
    timeFile=codecs.open("./time.txt","w","utf-8")
    for i in range(0,len(time_list)-1,1):
        timeFile.write(time_list[i]+"\n")
    timeFile.write(endtime)
    timeFile.close()
    delta1 = datetime.timedelta(days=-int(time_list[len(time_list)-3]))
    delta2 = datetime.timedelta(days=-int(time_list[len(time_list)-2]))
    current_time = datetime.datetime.now()
    event_time1 = (current_time + delta2).strftime('%Y-%m-%d %H:%M:%S')
    event_time2 = current_time.strftime('%Y-%m-%d %H:%M:%S')
    news_time1=(current_time+delta1).strftime('%Y-%m-%d ')+starttime
    news_time2=(current_time+delta1).strftime('%Y-%m-%d ')+endtime
    news_starttime = news_time1
    news_endtime =news_time2
    event_starttime =event_time1
    event_endtime =event_time2
    now=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    NewsFiles = ReadmongoData(news_starttime,news_endtime)#读取需要聚类的单篇新闻，返回单篇新闻的信息，以列表的形式存储，每一篇新闻是一个[{新闻1}，{新闻2}，.....]
    #WriteMongoEventData(NewsFiles)
    ClusterEventFiles = cluster(NewsFiles,clusterThreshold)#对单篇新闻进行聚类，返回聚类后的事件结果。同样是以列表的形式存储
    del NewsFiles
# ####分裂部分结束
    (EventFiles,totalEvent) = ReadMongoEventData(event_starttime,event_endtime)
    Numevent=len(EventFiles)
    NumOfEvent=totalEvent
    NumOfEvent=int(time.strftime("%Y%m%d%H", time.localtime())+"00000000")
    if len(EventFiles)==0:#未找到该窗口内的事件，直接将聚类的结果存储
        for line in ClusterEventFiles:
            line["id"]=NumOfEvent
            NumOfEvent +=1
        MergeEventFiles=ClusterEventFiles
        #WriteMongoEventData(MergeEventFiles)
    else:
        MergeEventFiles = merge(NumOfEvent,mergeThreshold,ClusterEventFiles,EventFiles)#如果是在合并完成后再进行分裂，在合并之后应该返
    del ClusterEventFiles
        # 回合并之后目前总的事件总数，便于分裂时事件ID的统计。事件是通过事件ID号进行关联，ID号十分重要
######事件合并完成，进行分裂#####
    number = NumOfEvent+len(MergeEventFiles)-Numevent
    TMP_cluster = list(MergeEventFiles)  #########有待考虑
    num=-1
    for line in MergeEventFiles:
        num +=1
        if line[u"count"] < 10:
            continue
        var = variance(line)
        # 进行判断，重新读取新闻表内的内容进行聚类
        #print var
        # if line[u"count"]>2000:
        #     line[u"closed"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #     TMP_cluster[num]=line
        #     continue
        if var > varThreshold or line["count"]>300:
            line[u"closed"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            closedID.append(line[u"id"])
            client = MongoClient(database_IP, dataport)
            db_name = dbname  # 数据库名
            db = client[db_name]
            connection = db[news_connection]  # 临时的新闻数据表，用于测试
            file_list = []
            for article in line[u"articles"]:
                for file in connection.find({"_id": article["id"]}):
                    file_list.append(file)
            step=0.1
            if line["count"]>300:
                step=0.2
            splitEvent = cluster(file_list,clusterThreshold+step)
            if len(splitEvent)>1:
                line[u"childrens"] = []
                line[u"closed"]=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
                for spevnet in splitEvent:
                    spevnet[u"id"] = number
                    line[u"childrens"].append(number)
                    spevnet[u"father"] = line[u"id"]
                    number += 1
                    TMP_cluster.append(spevnet)  #####循环的与增加的列表不是一个list主要是担心会出现循环迭代的情况，需要跟踪程序调试
    MergeEventFiles=TMP_cluster
    del TMP_cluster
    print "1:::;;;;;"
    ###进行事件信息抽取
    #对事件内的新闻进行排序,并且找出代表新闻,以及将事件内的关键词以及5W进行排序
    for event in MergeEventFiles:
        #if event[u"closed"]!=False:
            #continue
        if event[u"updated"]< now:
            continue
        event=sort(event)
    #计算窗口内的事件相关性与实体关系排序
    print "事件相关性计算"
    for eventi in MergeEventFiles:
        if eventi[u"closed"]!=False:
            continue
        if eventi[u"count"]<3:
            continue
        similarity_dict = {}
        event_narryi = array(eventi[u"eventVector"])
        for eventj in MergeEventFiles:
            if eventj[u"closed"] != False:
                continue
            if eventj[u"count"] <3:
                continue
            if eventi[u"id"]!=eventj[u"id"]:
                event_narryj=array(eventj[u"eventVector"])
                dp = dot(event_narryi, event_narryj)
                np = norm(event_narryi) * norm(event_narryj) + 0.0001
                sim = dp / np
                try:
                    similarity_dict[eventj[u"id"]] = sim
                except Exception, e:
                    print Exception, ":", e
                    print eventj[u"id"]
                    print type(eventj[u"id"])
            ####进行排序
        SortResult = sorted(similarity_dict.items(), key=lambda d: d[1], reverse=True)
        eventi[u"relatedEvents"]=[]
        for id, sim in SortResult:
              eventi[u"relatedEvents"].append({u"id":id,u"score":sim})
        del similarity_dict
    print "2::::::;;;;;;;"
        ###实体关系排序
#######事件信息抽取完成。提取的内容包括事件的代表新闻，实体的排序，事件的关键词抽取与排序以及when/where/who等信息的抽取
    #将处理的结果写入时间数据库中
    WriteMongoEventData(MergeEventFiles)
    time2 = time.time()
    print str(time2 - time1)
#####新增closedID部分
    client = MongoClient(database_IP, dataport)
    db_name = dbname  # 'NES'  # 数据库名
    db = client[db_name]
    connection = db[closedConnection]  # 临时的新闻数据表，用于测试
    closedFile={}
    closedFile[u"time"]=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    closedFile[u"closedId"]=closedID
    try:
        connection.insert(closedFile)
    except Exception, e:
        print Exception, ":", e
#if __name__ == "__main__":
#    timer = threading.Timer(1, clock)
#    timer.start()