from functools import wraps
import time

def timer(func):
    @wraps(func)
    def timed_func(*args, **kwargs):
        start = time.time_ns()
        val = func(*args, **kwargs)
        dt_ms = (time.time_ns() - start) / 1_000_000
        return dt_ms, val

    return timed_func