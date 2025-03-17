import time
from collections import defaultdict
import functools

prev_timing_dict = defaultdict(bool)
timing_dict = defaultdict(bool)
prev_counting_dict = defaultdict(bool)
counting_dict = defaultdict(bool)


def update_timing(key, dt, count=False, print_always=False):
    timing_dict[key] += dt
    if timing_dict[key] - prev_timing_dict[key] > 5 or print_always:
        print(key, timing_dict[key])
        prev_timing_dict[key] = timing_dict[key]

    if count:
        counting_dict[key] += 1
        if counting_dict[key] - prev_counting_dict[key] > 10 or print_always:
            print(key, counting_dict[key])
            prev_counting_dict[key] = counting_dict[key]


def timer(func=None, *, tag=None, count=False, print_always=False):
    def outer(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            out = f(*args, **kwargs)
            dt = time.time() - t0
            if tag is not None:
                update_timing(tag, dt, count, print_always)
                update_timing((tag, f.__name__), dt, count, print_always)
            else:
                update_timing(f.__name__, dt, count, print_always)
            return out
        return wrapper
    return outer if func is None else outer(func)


class Timer:
    def __init__(self, tag):
        self.t0 = self.tp = time.time()
        self.dt = 5
        self.tag = "\033[94m" + tag + "\033[0m"
        print("\033[1msetup timer", tag, "\033[0m")

    def __call__(self, tag=None, force=False):
        t = time.time()
        # if (t := time.time()) - self.tp > self.dt:
        tag = self.tag + ("" if tag is None else f"\033[96m {tag}\033[0m")
        if (t - self.tp > 0.01) or force:
            print(tag, t - self.tp)
        self.tp = t
