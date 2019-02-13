import time
import math
import numpy as np

class AveMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Timer(object):
    def __init__(self):
        self.start = time.time()

    def reset(self, t):
        self.start = t

    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self):
        now = time.time()
        s = now - self.start
        return '%s' % (self.asMinutes(s))