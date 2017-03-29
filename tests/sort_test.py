import pyopencl as cl
import numpy as np
from timeit import timeit_repeat
from pyopencl.algorithm import RadixSort
from pyopencl.bitonic_sort import BitonicSort
from pyopencl import clrandom
from pyopencl.scan import GenericScanKernel

device = cl.get_platforms()[1].get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
reps = 64


@timeit_repeat(reps)
def test_radix_speed(buff, sorter):
    sorter(buff)[1].wait()


@timeit_repeat(reps)
def test_bitonic_speed(buff, sorter):
    sorter(buff)[1].wait()


@timeit_repeat(reps)
def test_numpy_speed(buff):
    np.sort(buff)


from collections import defaultdict

times = defaultdict(list)
dtype = np.int32
xs = range(5, 32)
bs = BitonicSort(ctx)
rs = RadixSort(ctx, "int *ary", key_expr="ary[i]", sort_arg_names=["ary"],
               scan_kernel=GenericScanKernel)
for size in xs:
    print("running size=2^{} = {}".format(size, 2 ** size))
    s = clrandom.rand(queue, (2 ** size,), dtype, a=0, b=2 ** 16)
    times['bitonic'].append(test_bitonic_speed(s, bs).microseconds / 1000000.)
    times['radix'].append(test_radix_speed(s, rs).microseconds / 1000000.)
    times['numpy'].append(test_numpy_speed(s.get().copy()).microseconds / 1000000.)

print("\t".join(["Size"] + times.keys()))
for idx, s in enumerate(xs):
    print("\t".join(["2^"+str(s)]+[str(times[k][idx]) for k in times.keys()]))

font = {'size': 30}
import matplotlib
import matplotlib.pyplot as plt
from pluck import pluck

matplotlib.rc('font', **font)
fig, ax = plt.subplots()
points = ['r^', 'gv', 'r*']
for name, time in times.items():
    plt.plot(xs, time, points.pop(), label=name)
plt.legend(prop={'size': 23})
fig.subplots_adjust(bottom=0.15)

plt.grid()
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='black', linestyle='dotted')
plt.title("Comparison of sorts", y=1.03)
plt.ylabel("Average Running Time over {} repeats(s)".format(reps))
plt.xlabel("Array {} Size".format(dtype))
plt.xticks(rotation='vertical')
plt.show()
