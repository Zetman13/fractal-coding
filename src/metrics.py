import time
import numpy as np


def psnr(x1, x2):
    return 10*np.log10(255**2/np.square(x1-x2).mean())


def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(f"{func.__name__}: {t2 - t1}")
        return res
    return wrapper
