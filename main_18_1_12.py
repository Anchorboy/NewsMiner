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

# def test(args):
#     IP_PORT = args.ip_port
#
#     news_reader = NewsReader(uri=IP_PORT)
#     event_reader = EventReader(uri=IP_PORT, event_name="test_en_event1")
#     event_reader.remove_collection()
#
#     start_time_t = "2016-07-20 16:00:00"
#     end_time_t = "2016-08-30 18:00:00"
#     day_diff = 86400
#     day_window = args.day_window
#
#     func = Function()
#     print "test start"
#     # start time
#     start_time_ts = func.time2time_stamp(start_time_t)
#     current_time_start_ts = start_time_ts
#     current_time_end_ts = start_time_ts + day_window * day_diff
#     end_time_ts = func.time2time_stamp(end_time_t)
#
#     while current_time_end_ts < end_time_ts:
#         current_time_start_t = func.time_stamp2time(current_time_start_ts)
#         current_time_end_t = func.time_stamp2time(current_time_end_ts)
#         print "start", current_time_start_t, "end", current_time_end_t
#         news_list = news_reader.query_many_by_time(start_time=current_time_start_t, end_time=current_time_end_t)
#         time_info = ((current_time_start_ts, current_time_start_t), (current_time_end_ts, current_time_end_t))
#
#         # clustering ----------------------------
#         clustering = Model(dim=args.dimension,
#                            sim_thres=args.sim,
#                            merge_sim_thres=args.merge_sim,
#                            subevent_sim_thres=args.sub_sim,
#                            class_file=args.class_file,
#                            news_reader=news_reader,
#                            event_reader=event_reader)
#         clustering.main(news_list=news_list, time_info=time_info)
#         del clustering
#         # ----------------------------------------
#         current_time_start_ts += day_window * day_diff
#         current_time_end_ts += day_window * day_diff

def main(args):
    config = Config(args)

    print "reading news"
    news_reader = NewsReader(uri=config.ip_port)
    event_reader = EventReader(uri=config.ip_port)
    news_list = news_reader.query_many_by_time(start_time=config.start_time_t, end_time=config.end_time_t)
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

    # cmd_parser = subparsers.add_parser('debug', help='debug: running test()')
    # cmd_parser.add_argument("-ip", "--ip_port", default="10.1.1.46:27017", help="IP & port. default=10.1.1.46:27017")
    # cmd_parser.add_argument("-e", default="test_en_event", help="Event collection name. default=test_en_event")
    # cmd_parser.add_argument("-dim", "--dimension", default=dim, type=int, help="Vector dimension. default=2200")
    # cmd_parser.add_argument("-f", "--class_file", default="utils/" + str(dim) + ".txt",
    #                         help="Input class file. default=utils/2200.txt")
    # cmd_parser.add_argument("-d", '--day_window', default=1, type=int, help="Day window to clustering news. default=1")
    # cmd_parser.add_argument("-s", '--sim', default=0.7, type=float, help="Similarity threshold. default=0.7")
    # cmd_parser.add_argument("-ss", '--sub_sim', default=0.75, type=float,
    #                         help="Subevent similarity threshold. default=0.75")
    # cmd_parser.add_argument("-ms", '--merge_sim', default=0.75, type=float,
    #                         help="Merge similarity threshold. default=0.75")
    # cmd_parser.set_defaults(func=test)

    cmd_parser = subparsers.add_parser('main', help='running main()')
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
    cmd_parser.set_defaults(func=main)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)