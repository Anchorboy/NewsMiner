import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
from reader import NewsReader, EventReader
from clustering_v3 import Model
from utils.function import Function
from datetime import *
import time
import argparse

def test():
    dim = 2200
    class_file = "model/" + str(dim) + ".txt"
    sim = 0.7
    merge_sim = 0.7
    sub_sim = 0.7
    IP_PORT = "10.1.1.46:27017"

    news_reader = NewsReader(uri=IP_PORT)
    event_reader = EventReader(uri=IP_PORT)
    event_reader.remove_collection()
    # clustering = Clustering(sim_thres=sim, merge_sim_thres=merge_sim, subevent_sim_thres=sub_sim, dim=dim,
    #                         class_file=str(dim) + ".txt", news_reader=news_reader, event_reader=event_reader)
    start_time_t = "2016-07-25 16:00:00"
    end_time_t = "2016-07-28 18:00:00"
    day_diff = 86400
    day_window = 1

    func = Function()
    print "test start"
    # start time
    start_time_ts = func.time2time_stamp(start_time_t)
    current_time_start_ts = start_time_ts
    current_time_end_ts = start_time_ts + day_window * day_diff
    end_time_ts = func.time2time_stamp(end_time_t)

    while current_time_end_ts < end_time_ts:
        current_time_start_t = func.time_stamp2time(current_time_start_ts)
        current_time_end_t = func.time_stamp2time(current_time_end_ts)
        print "start", current_time_start_t, "end", current_time_end_t
        news_list = news_reader.query_many_by_time(start_time=current_time_start_t, end_time=current_time_end_t)
        time_info = ((current_time_start_ts, current_time_start_t), (current_time_end_ts, current_time_end_t))

        # clustering ----------------------------
        clustering = Model(sim_thres=sim, merge_sim_thres=merge_sim, subevent_sim_thres=sub_sim, dim=dim,
                           class_file=class_file, news_reader=news_reader, event_reader=event_reader)
        clustering.main(news_list=news_list, time_info=time_info)
        # ----------------------------------------
        current_time_start_ts += day_window * day_diff
        current_time_end_ts += day_window * day_diff

def main(args):
    func = Function()
    current_path = os.path.abspath('.')
    log_dir = os.path.join(current_path, 'log')

    print "get previous time"
    now = datetime.now()
    end_time_t = now.strftime("%Y-%m-%d %H:%M:%S")
    start_time_t = end_time_t
    if os.path.exists(log_dir) and False:
        log_file = os.path.join(log_dir, 'log.json')
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log = json.loads(f.read())
            start_time_t = log['end']
    else:
        day_diff = 86400
        day_window = args.d
        start_time_ts = float(int(time.time() - day_diff * day_window))
        start_time_t = func.time_stamp2time(start_time_ts)
    assert(not start_time_t==end_time_t)
    time_info = func.generate_timeinfo(start_t=start_time_t, end_t=end_time_t)
    print time_info
    print "---------------"

    print "reading news"
    dim = 2200
    class_file = "model/" + str(dim) + ".txt"
    IP_PORT = args.ip
    news_reader = NewsReader(uri=IP_PORT)
    event_reader = EventReader(uri=IP_PORT)
    news_list = news_reader.query_many_by_time(start_time=start_time_t, end_time=end_time_t)
    print news_list.count()
    cnt = 0
    for news in news_list:
        if 'stemContent' not in news:
            cnt += 1
    print cnt
    print "---------------"

    print "start clustering"
    clustering = Model(dim=args.dim,
                       sim_thres=args.s,
                       merge_sim_thres=args.ms,
                       subevent_sim_thres=args.ss,
                       class_file=args.f,
                       news_reader=news_reader,
                       event_reader=event_reader)
    clustering.main(news_list=news_list, time_info=time_info)
    print "---------------"


if __name__ == "__main__":
    dim = 2200
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", default="10.1.1.46:27017")
    parser.add_argument("-debug", default=0, type=int)
    parser.add_argument("-dim", default=dim, type=int)
    parser.add_argument("-f", default="model/" + str(dim) + ".txt")
    parser.add_argument("-d", default=1, type=int)
    parser.add_argument("-s", default=0.7, type=float)
    parser.add_argument("-ss", default=0.7, type=float)
    parser.add_argument("-ms", default=0.75, type=float)
    args = parser.parse_args()
    if args.debug == 1:
        test()
    else:
        main(args)
