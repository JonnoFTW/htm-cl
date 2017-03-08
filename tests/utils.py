from __future__ import print_function
import inspect
import os


def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith('pydevd.py'):
            return True
    return False


def follow_link(fname):
    print(fname, end=" ")
    if os.path.islink(fname):
        print("\n\t->", end="")
        next_l = os.readlink(fname)
        d = '/'.join(fname.split('/')[:-1])
        if os.path.isabs(next_l):
            follow_link(next_l)
        else:
            follow_link(d + '/' + next_l)
    else:
        print()


if __name__ == "main":

    out = os.popen('locate libOpenCL.so').readlines()
    for i in out:
        i = i.strip()
        follow_link(i)
    print(os.environ['PYTHONPATH'].split(os.pathsep))
    print(os.environ['LD_LIBRARY_PATH'])
