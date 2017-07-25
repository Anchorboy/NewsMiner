#! /usr/bin/python
# -*- coding:utf-8 -*-
import json
from datetime import *
import time
from pymongo import MongoClient

class Reader():
    def __init__(self, uri):
        self._db_name = "NES"
        self._news_collection_name = "en_news"
        self._event_collection_name = "en_event"
        self.client = MongoClient(uri)
        self.db = self.client[self._db_name]

    def parse_uri(self, host, username, pswd):
        uri = "mongodb://" + username + ":" + pswd + "@" + host + "?authSource=source"
        return uri

    def time2time_stamp(self, t):
        timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
        timeStamp = float(time.mktime(timeArray))
        return timeStamp

    def time_stamp2time(self, t):
        return str(datetime.fromtimestamp(t))

    def remove_collection(self):
        pass

    def insert_item(self, item):
        pass

    def save_item(self, item):
        pass

    def query_many_by_time(self, start_time, end_time):
        pass

    def query_one_by_item(self, item):
        pass

    def query_many_by_item(self, item):
        pass

    def read_txt(self, filename):
        with open(filename, "r") as f:
            return json.loads(f.read())

class NewsReader(Reader):
    def __init__(self, uri):
        Reader.__init__(self, uri=uri)
        self.init_mongoDB()

    def init_mongoDB(self):
        """
        初始化mongoDB, 包含client, db, collection
        :param uri: 輸入mongoDB的uri位址
        :return: none
        """
        self.news_collection = self.db[self._news_collection_name]
        print self.db
        print self.news_collection

    def remove_collection(self):
        self.news_collection.remove()

    def insert_item(self, item):
        result = self.news_collection.insert(item)
        return result

    def save_item(self, item):
        result = self.news_collection.save(item)
        return result

    def query_many_by_time(self, start_time, end_time):
        """
        尋找mongoDB news collection中符合時間段內的新聞
        :param start_time: 開始時間 (上次查詢後最後時間)
        :param end_time: 結束時間 (time.time() 現在運行時間)
        :return: result: 查詢結果
        """
        result = self.news_collection.find({"crawlTime": {"$gt": start_time, "$lt": end_time}})
        # for i in result:
        #     print i
        return result

    def query_many_by_item(self, item):
        """
        根據提供的item尋找mongoDB news collection中符合的新聞
        :param item: 查詢的item條件, dict
        :return: result: 查詢結果
        """
        result = self.news_collection.find(item)
        # for i in result:
        #     print i
        return result

    def query_one_by_item(self, item):
        """
        根據提供的item尋找mongoDB news collection中符合的新聞
        :param item: 查詢的item條件, dict
        :return: result: 查詢結果
        """
        result = self.news_collection.find_one(item)
        # for i in result:
        #     print i
        return result

class EventReader(Reader):
    def __init__(self, uri):
        Reader.__init__(self, uri=uri)
        self.init_mongoDB()
        self.day_diff = 86400
        self.window = 10

    def init_mongoDB(self):
        """
        初始化mongoDB, 包含client, db, collection
        :param uri: 輸入mongoDB的uri位址
        :return: none
        """
        self.event_collection = self.db[self._event_collection_name]
        print self.db
        print self.event_collection

    def remove_collection(self):
        self.event_collection.remove()

    def test(self):
        result = self.event_collection.find()
        for i in result:
            print i

    def insert_item(self, item):
        result = self.event_collection.insert(item)
        return result

    def save_item(self, item):
        result = self.event_collection.save(item)
        return result

    def create_event_id(self, t):
        """

        :param t: 2016-06-20 17:00:00
        :return: eid: 20170620170000
        """
        eid = t.replace('-', '')
        eid = eid.replace(':', '')
        eid = eid.replace(' ', '')
        return eid

    def query_recent_events_by_time(self, t):
        last_time = int(Reader.time2time_stamp(self, t) - self.day_diff * self.window)
        last_time = Reader.time_stamp2time(self, last_time)
        return self.query_many_by_time(start_time=last_time, end_time=t)

    def query_recent_events_by_timestamp(self, ts):
        last_time = int(ts - self.day_diff * self.window)
        last_time = Reader.time_stamp2time(self, last_time)
        return self.query_many_by_time(start_time=last_time, end_time=ts)

    def query_many_by_time(self, start_time, end_time):
        """
        尋找mongoDB event collection中符合時間段內的新聞
        :param start_time: 開始時間 (上次查詢後最後時間)
        :param end_time: 結束時間 (time.time() 現在運行時間)
        :return: result: 查詢結果
        """
        result = self.event_collection.find({"updated": {"$gt": start_time, "$lt": end_time}})
        # for i in result:
        #     print i
        return result

    def query_many_by_item(self, item):
        """
        根據提供的item尋找mongoDB news collection中符合的新聞
        :param item: 查詢的item條件, dict
        :return: result: 查詢結果
        """
        result = self.event_collection.find(item)
        # for i in result:
        #     print i
        return result

    def query_one_by_item(self, item):
        """
        根據提供的item尋找mongoDB news collection中符合的新聞
        :param item: 查詢的item條件, dict
        :return: result: 查詢結果
        """
        result = self.event_collection.find_one(item)
        # for i in result:
        #     print i
        return result

def visualize():
    pass

def test_news():
    IP_PORT = "10.1.1.46:27017"
    news_reader = NewsReader(uri=IP_PORT)
    print "query news from 2016-07-25 16:00:00 to 2016-07-27 16:00:00"
    news_list = news_reader.query_many_by_time(start_time="2017-07-20 16:00:00", end_time="2017-07-21 16:00:00")
    for i in news_list:
        print i['title'].decode('utf-8', errors="ignore")
        print "-----------------------"

def test_event():
    # news_reader = NewsReader(uri='localhost')
    # news_list = news_reader.query_mongoDB_by_time(start_time="2016-11-20 16:00:00", end_time="2016-11-20 18:00:00")
    IP_PORT = "10.1.1.46:27017"
    event_reader = EventReader(uri=IP_PORT)
    # result = event_reader.query_recent_events_by_time(t="2016-07-30 16:00:00")
    # ts = event_reader.time2time_stamp(t="2016-07-27 16:00:00")
    # result = event_reader.query_recent_events_by_time("2016-07-31 16:00:00")
    result = event_reader.query_many_by_time("2016-07-25 16:00:00", "2016-07-27 16:00:00")
    for i in result:
        # print i['articles']
        # print i['relatedEvents']
        count = i['count']
        if count > 20:
            print i['count']
            print i['articles']
            print i['keynews']
            print i['keywords']
            print i['persons']
            print "-----------------------"

if __name__ == "__main__":
    test_news()
    test_event()
