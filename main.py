import sys
from reader import NewsReader, EventReader
from clustering_v3 import Clustering
from datetime import *
import time

def time2time_stamp(t):
    timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
    timeStamp = float(time.mktime(timeArray))
    return timeStamp

def time_stamp2time(t):
    return str(datetime.fromtimestamp(t))

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    dim = 2200
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
    end_time_t = "2016-07-30 18:00:00"
    day_diff = 86400
    day_window = 1

    # start time
    start_time_ts = time2time_stamp(start_time_t)
    current_time_start_ts = start_time_ts
    current_time_end_ts = start_time_ts + day_window * day_diff
    end_time_ts = time2time_stamp(end_time_t)

    while current_time_end_ts < end_time_ts:
        current_time_start_t = time_stamp2time(current_time_start_ts)
        current_time_end_t = time_stamp2time(current_time_end_ts)
        print "start", current_time_start_t, "end", current_time_end_t
        news_list = news_reader.query_many_by_time(start_time=current_time_start_t, end_time=current_time_end_t)
        time_info = ((current_time_start_ts, current_time_start_t), (current_time_end_ts, current_time_end_t))

        #clustering ----------------------------
        clustering = Clustering(sim_thres=sim, merge_sim_thres=merge_sim, subevent_sim_thres=sub_sim, dim=dim,
                                class_file=str(dim) + ".txt", news_reader=news_reader, event_reader=event_reader)
        clustering.main(news_list=news_list, time_info=time_info)
        # ----------------------------------------
        current_time_start_ts += day_window * day_diff
        current_time_end_ts += day_window * day_diff

    current_time_end_ts = end_time_ts
    current_time_start_t = time_stamp2time(current_time_start_ts)
    current_time_end_t = time_stamp2time(current_time_end_ts)
    print "start", current_time_start_t, "end", current_time_end_t
    news_list = news_reader.query_many_by_time(start_time=current_time_start_t, end_time=current_time_end_t)
    time_info = ((current_time_start_ts, current_time_start_t), (current_time_end_ts, current_time_end_t))

    clustering = Clustering(sim_thres=sim, merge_sim_thres=merge_sim, subevent_sim_thres=sub_sim, dim=dim,
                            class_file=str(dim) + ".txt", news_reader=news_reader, event_reader=event_reader)
    clustering.main(news_list=news_list, time_info=time_info)
