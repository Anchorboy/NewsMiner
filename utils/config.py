import os
import json
from function import Function
from datetime import *
import time

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # it can be used to replace args
    ip_port = "10.1.1.46:27017"
    dim = 2200
    day_window = 1
    sim = 0.7
    sub_sim = 0.75
    merge_sim = 0.75

    def __init__(self, args):
        func = Function()
        log_dir = os.path.join('log')
        log_file = os.path.join(log_dir, 'log.json')
        print "get previous time"
        if not args.end_time_t or not args.start_time_t:
            now = datetime.now()
            end_time_t = now.strftime("%Y-%m-%d %H:%M:%S")
            if os.path.exists(log_dir) and os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_dict = json.loads(f.read())
                start_time_t = log_dict['end']
            else:
                day_diff = 86400
                day_window = args.day_window
                start_time_ts = float(int(time.time() - day_diff * day_window))
                start_time_t = func.time_stamp2time(start_time_ts)
        else:
            end_time_t = args.end_time_t
            start_time_t = args.start_time_t

        assert (not start_time_t == end_time_t)
        time_info = func.generate_timeinfo(start_t=start_time_t, end_t=end_time_t)
        print time_info

        # --- time here
        self.start_time_t = start_time_t
        self.end_time_t = end_time_t
        self.time_info = time_info

        # --- args here
        self.ip_port = args.ip_port
        self.dim = args.dimension
        self.class_file = args.class_file
        self.day_window = args.day_window
        self.sim_thres = args.sim
        self.subevent_sim_thres = args.sub_sim
        self.merge_sim_thres = args.merge_sim

        self.output_path = os.path.join("Output",
                                        's{}ms{}sub{}dim{}'.format(self.sim_thres, self.merge_sim_thres, self.subevent_sim_thres, self.dim))