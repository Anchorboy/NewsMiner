import os
import re
from datetime import *
import time
from header import get_news_json
from pymongo import MongoClient
from nltk.stem.porter import PorterStemmer

sw_file = open("stopwords_en.txt", 'r')
stopwords = list()
with sw_file as f:
    for line in f:
        stopwords.append(line.strip())

porter_stemmer = PorterStemmer()
pattern = re.compile("[a-zA-Z]+")

def time2time_stamp(t):
    timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
    timeStamp = float(time.mktime(timeArray))
    return timeStamp

def time_stamp2time(t):
    return datetime.fromtimestamp(t)

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
                except IndexError:
                    print match.group()
    return lower_content, stem_content

def insert_news(input_base, collection):
    count = 0
    for year_dir in os.listdir(input_base):
        print year_dir
        year_path = os.path.join(input_base, year_dir)

        for month_dir in os.listdir(year_path):
            print month_dir
            month_path = os.path.join(year_path, month_dir)

            for date_dir in os.listdir(month_path):
                print date_dir
                count = 0
                dir = os.path.join(month_path, date_dir)
                t = year_dir + "-" + month_dir + "-" + date_dir + " 17:00:00"

                for news_file in os.listdir(dir):
                    filepath = os.path.join(dir, news_file)
                    news_json = get_news_json()

                    title = ""
                    with open(filepath, "r") as f:
                        for line in f:
                            l = line.strip().split(":", 1)
                            attribute = l[0]
                            value = l[1]
                            if attribute == "id":
                                news_json['_id'] = year_dir + month_dir + date_dir + "N" + str(count); count += 1
                            elif attribute == "URL":
                                news_json['url'] = value
                            elif attribute == "category":
                                news_json['category'] = value
                            elif attribute == "Content":
                                news_json['content'] = value; news_json['lowerContent'], news_json['stemContent'] = preprocess(title + " " + value)
                            elif attribute == "Journalist":
                                news_json['publisher'] = value
                            elif attribute == "Image":
                                images = value.split(); news_json['image'] = images
                            elif attribute == "Title":
                                news_json['title'] = value
                    news_json['crawlTime'] = t
                    collection.save(news_json)

    print "insert", count, "news"

if __name__ == "__main__":
    IP_PORT = "10.1.1.46:27017"
    client = MongoClient("10.1.1.46:27017")
    database_name = "NES"
    collection_name = "en_news"
    db = client[database_name]
    news_collection = db[collection_name]

    # print __news__
    # collection.remove({"crawlTime":{"$gt": "2016-10-01 00:00:00", "$lt": "2016-11-01 00:00:00"}})

    current_base = os.path.abspath('.')
    input_base = "D:/TestData/news/"
    insert_news(input_base, news_collection)