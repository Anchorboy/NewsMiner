import os
import re
from datetime import *
import time
from header import *
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
    for token in tokens:
        match = re.match(pattern, token)
        # regular expression
        if match:
            # remove stop words
            if match.group().lower() not in stopwords:
                # stemming
                try:
                    stem = porter_stemmer.stem(match.group().lower())
                    stem_content += stem + " "
                except IndexError:
                    print match.group()
    return stem_content

def insert_news(input_base, collection):
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
                    news = Header.__news__.copy()
                    duplicate = False
                    title = ""
                    with open(filepath, "r") as f:
                        for line in f:
                            l = line.strip().split(":", 1)
                            attribute = l[0]
                            value = l[1]
                            if attribute == "id":
                                news['_id'] = year_dir + month_dir + date_dir + "N" + str(count); count += 1
                            elif attribute == "URL":
                                news['url'] = value
                            elif attribute == "category":
                                news['category'] = value
                            elif attribute == "Content":
                                news['content'] = value; news['stemContent'] = preprocess(title + " " + value)
                            elif attribute == "Journalist":
                                news['publisher'] = value
                            elif attribute == "Image":
                                images = value.split(); news['image'] = images
                            elif attribute == "Title":
                                result = collection.find_one({'title': value})
                                news['title'] = value
                                title = value
                                if result:
                                    duplicate = True
                                    break
                    news['crawlTime'] = t
                    if not duplicate:
                        collection.insert(news)

if __name__ == "__main__":
    client = MongoClient('localhost')
    database_name = "NES"
    collection_name = "news"
    db = client[database_name]
    collection = db[collection_name]
    # collection.remove({"crawlTime":{"$gt": "2016-10-01 00:00:00", "$lt": "2016-11-01 00:00:00"}})

    # current_base = os.path.abspath('.')
    # input_base = "D:/TestData/11-01/"
    # insert_news(input_base, collection)