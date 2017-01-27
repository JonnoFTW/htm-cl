import functools
from datetime import datetime, timedelta


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = datetime.now()
        func(*args, **kwargs)
        elapsedTime = datetime.now() - startTime
        print('function [{}] finished in {}'.format(
            func.__name__, elapsedTime))

    return newfunc


def timeit_repeat(repeats=100):
    def _timeit(func):
        @functools.wraps(func)
        def newfunc(*args, **kwargs):
            times = []
            params = list(args) + map(lambda x: "{}={}".format(x[0], repr(x[1])), kwargs.items())
            print(
                "Repeating [{}{}] {} times....".format(func.__name__, '(' + ', '.join(map(str, params)) + ')', repeats))
            for i in xrange(repeats):
                start_time = datetime.now()
                func(*args, **kwargs)
                elapsed_time = datetime.now() - start_time
                times.append(elapsed_time)
            avg = sum(times, timedelta(0)) / len(times)
            print('Function [{}] finished in average of {}'.format(
                func.__name__, avg))

        return newfunc

    return _timeit


if __name__ == "__main__":
    import time


    @timeit_repeat(10)
    def t1(a, b=5, c=89):
        time.sleep(.5)
        return sum(map(lambda x: x ** 64, [a] * 1024))


    t1(5, b='6', c=5)
