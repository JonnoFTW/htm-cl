from future.builtins import range
import functools
from datetime import datetime, timedelta


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = datetime.now()
        out = func(*args, **kwargs)
        elapsedTime = datetime.now() - startTime
        print('function [{}] finished in {}\n'.format(
            func.__name__, elapsedTime))
        return elapsedTime

    return newfunc


def timeit_repeat(repeats=100, verbose=False):
    def _timeit(func):
        @functools.wraps(func)
        def newfunc(*args, **kwargs):
            times = []
            params = list(args) + map(lambda x: "{}={}".format(x[0], repr(x[1])), kwargs.items())
            if verbose:
                print(
                    "Repeating [{}{}] {} times....".format(func.__name__, '(' + ', '.join(map(str, params)) + ')',
                                                           repeats))
            for _ in range(repeats):
                start_time = datetime.now()
                func(*args, **kwargs)
                elapsed_time = datetime.now() - start_time
                times.append(elapsed_time)
            avg = sum(times, timedelta(0)) / len(times)
            print('Function [{}] finished in average of {}\n'.format(
                func.__name__, avg))
            return avg

        return newfunc

    return _timeit
