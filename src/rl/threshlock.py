from threading import Lock, Condition


class ThreshLock():
    """
    An implementation of a threshold lock. A thread calling wait will wait until
    a a given threshold of threads are all waiting, and then all threads are
    released.
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.waiting = 0
        self.m = Lock()
        self.cv = Condition(self.m)

    def wait(self):
        self.m.acquire()
        self.waiting += 1
        n = self.waiting
        if (self.waiting == self.threshold):
            self.cv.notify(self.threshold-1)
            self.waiting = 0
            self.m.release()
        else:
            self.cv.wait()
            self.m.release()
        return n
