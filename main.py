import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
from utils.reader import NewsReader, EventReader
from utils.function import Function
from utils.config import Config
from model import Model
from datetime import *
import time
import argparse

def debug(args):
    args.start_time_t = "2018-01-30 00:00:00"
    args.end_time_t = "2018-02-01 17:00:00"
    config = Config(args)

    print "reading news"
    news_reader = NewsReader(uri=config.ip_port, news_name='english_news')
    event_reader = EventReader(uri=config.ip_port, event_name='english_event', window=config.event_day_window)
    event_reader.remove_collection()
    
    start_time_t = args.start_time_t
    end_time_t = args.end_time_t
    print start_time_t, end_time_t
    day_window = args.day_window

    func = Function()
    print "test start"
    # start time
    start_time = func.time_string2time(start_time_t)
    end_time = func.time_string2time(end_time_t)
    cur_start_time = start_time
    cur_end_time = start_time + timedelta(days=day_window)

    while cur_start_time <= end_time:
        cur_start_time_t = func.time2time_string(cur_start_time)
        cur_end_time_t = func.time2time_string(cur_end_time)
        print "start", cur_start_time_t, "end", cur_end_time_t
        news_list = news_reader.query_many_by_time(start_time=cur_start_time_t, end_time=cur_end_time_t)
        time_info = (cur_start_time_t, cur_end_time_t)

        # clustering ----------------------------
        model = Model(config=config,
                      news_reader=news_reader,
                      event_reader=event_reader)
        model.run(news_list=news_list, time_info=time_info)
        # ----------------------------------------
        cur_start_time = cur_start_time + timedelta(days=day_window)
        cur_end_time = cur_end_time + timedelta(days=day_window)
        if cur_end_time > end_time:
            cur_end_time = end_time
        print "---------------"


def main(args):
    config = Config(args)

    print "reading news"
    news_reader = NewsReader(uri=config.ip_port)
    event_reader = EventReader(uri=config.ip_port, window=config.event_day_window)
    start_time_t, end_time_t = config.time_info
    print start_time_t, end_time_t
    news_list = news_reader.query_many_by_time(start_time=start_time_t, end_time=end_time_t)
    print "---------------"

    print "start clustering"
    clustering = Model(config=config,
                       news_reader=news_reader,
                       event_reader=event_reader)
    clustering.run(news_list=news_list, time_info=config.time_info)
    print "---------------"


if __name__ == "__main__":
    dim = 2200
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    cmd_parser = subparsers.add_parser('debug', help='debug: running test()')
    cmd_parser.add_argument("-ip", "--ip_port", default="10.1.1.46:27017", help="IP & port. default=10.1.1.46:27017")
    cmd_parser.add_argument("-dim", "--dimension", default=dim, type=int, help="Vector dimension. default=2200")
    cmd_parser.add_argument("-f", "--class_file", default="utils/" + str(dim) + ".txt",
                            help="Input class file. default=utils/2200.txt")
    cmd_parser.add_argument("-d", '--day_window', default=1, type=int, help="Day window to clustering news. default=1")
    cmd_parser.add_argument("-s", '--sim', default=0.7, type=float, help="Similarity threshold. default=0.7")
    cmd_parser.add_argument("-ss", '--sub_sim', default=0.75, type=float,
                            help="Subevent similarity threshold. default=0.75")
    cmd_parser.add_argument("-ms", '--merge_sim', default=0.75, type=float,
                            help="Merge similarity threshold. default=0.75")
    cmd_parser.add_argument("-st", '--start_time_t', type=str,
                            help="fomat 2018-01-01 17:00:00")
    cmd_parser.add_argument("-et", '--end_time_t', type=str,
                            help="fomat 2018-01-01 17:00:00")
    cmd_parser.set_defaults(func=debug)

    cmd_parser = subparsers.add_parser('main', help='running main()')
    cmd_parser.add_argument("-is_test", default=False, type=bool, help="test")
    cmd_parser.add_argument("-ip", "--ip_port", default="10.1.1.46:27017", help="IP & port. default=10.1.1.46:27017")
    cmd_parser.add_argument("-dim", "--dimension", default=dim, type=int, help="Vector dimension. default=2200")
    cmd_parser.add_argument("-f", "--class_file", default="utils/" + str(dim) + ".txt",
                            help="Input class file. default=utils/2200.txt")
    cmd_parser.add_argument("-d", '--day_window', default=1, type=int, help="Day window to clustering news. default=1")
    cmd_parser.add_argument("-s", '--sim', default=0.7, type=float, help="Similarity threshold. default=0.7")
    cmd_parser.add_argument("-ss", '--sub_sim', default=0.75, type=float,
                            help="Subevent similarity threshold. default=0.75")
    cmd_parser.add_argument("-ms", '--merge_sim', default=0.75, type=float,
                            help="Merge similarity threshold. default=0.75")
    cmd_parser.add_argument("-st", '--start_time_t', type=str,
                            help="fomat 2018-01-01 17:00:00")
    cmd_parser.add_argument("-et", '--end_time_t', type=str,
                            help="fomat 2018-01-01 17:00:00")
    cmd_parser.set_defaults(func=main)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
