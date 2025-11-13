import time
from collections import defaultdict
import functools

prev_timing_dict = defaultdict(bool)
timing_dict = defaultdict(bool)
prev_counting_dict = defaultdict(bool)
counting_dict = defaultdict(bool)


def update_timing(key, dt, min_dt=5, count=False, min_count=10):
    timing_dict[key] += dt
    if timing_dict[key] - prev_timing_dict[key] > min_dt:
        print(key, timing_dict[key])
        prev_timing_dict[key] = timing_dict[key]

    if count:
        counting_dict[key] += 1
        if counting_dict[key] - prev_counting_dict[key] > min_count:
            print(key, counting_dict[key])
            prev_counting_dict[key] = counting_dict[key]


def timer(func=None, *, tag=None, min_dt=5, count=False, min_count=10):
    """
    decorator to time a function call.
    Total time is printed to terminal (time per call X number of calls).
    Several functions can be grouped by providing a tag.
    Also the number of calls can be printed if *count=True*
    The minimum time / number of calls to trigger a print can also be specified

    @timer
    def my_func():
        ...

    OUT: my_func <time if time>

    @timer(tag="tag")
    def my_func():
        ...

    OUT: tag <time> (sum of all functions with this tag)
    OUT: tag my_func <time>


    """
    wrap_kwargs = dict(minD_dt=min_dt, count=count, min_count=min_count)
    def outer(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            out = f(*args, **kwargs)
            dt = time.time() - t0
            if tag is not None:
                update_timing(tag, dt, **wrap_kwargs)
                update_timing((tag, f.__name__), dt, **wrap_kwargs)
            else:
                update_timing(f.__name__, dt, **wrap_kwargs)
            return out
        return wrapper
    return outer if func is None else outer(func)


tzero = time.time()
time_dct = defaultdict(lambda: 0.)
time_dct_prev = defaultdict(lambda: 0.)


class Timer:
    """
    Class to create timer objects.
    Example:

    tmr = Timer("my_timer")  # at t0
    ...
    tmr("checkpoint 1") # at t1
    OUT: my_timer checkpoint 1 (t1 - t0) ...
    ...
    tmr("checkpoint 2") # at t2
    OUT: my_timer checkpoint 2 (t2 - t1) ...

    Also the accumulated sum of the reported time intervals
    (per checkpoint tag) is reported, together with the total time lapsed
    (and the fraction)

    """
    def __init__(self, tag, report_interval=1):
        self.previous_call_time = self.time()
        self.report_interval = report_interval
        self.tag = "\033[94m" + tag + "\033[0m"

    def __call__(self, tag=None, force=False, silent=False):
        t = self.time()
        ftag = self.tag + ("" if tag is None else f"\033[96m {tag}\033[0m")
        time_dct[ftag] += (dt := t - self.previous_call_time)
        if not silent and (force or (time_dct[ftag] - time_dct_prev[ftag] > self.report_interval)):
            print(ftag, round(dt, 4), "s,", *self.total_time(tag, t))
            time_dct_prev[ftag] = time_dct[ftag]
        self.previous_call_time = t
        return dt

    def total_time(self, tag=None, t=None):
        tag = self.tag + ("" if tag is None else f"\033[96m {tag}\033[0m")
        t = t or self.time()
        total_time = t - tzero
        return f"{time_dct[tag] / 60:.2f}min", f"{total_time / 60:2f}min", f"{time_dct[tag] / total_time * 100:.2f}%"

    @classmethod
    def time(cls):
        return time.time()
