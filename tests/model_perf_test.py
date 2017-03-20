from __future__ import print_function
import pyopencl as cl

from collections import defaultdict
import numpy as np
import os
import os.path
from copy import deepcopy
from datetime import datetime
import pyprind

context = cl.Context([cl.get_platforms()[1].get_devices()[0]])

from tests.timeit import timeit_repeat
from src import Model as CLModel
from tests.models import NUModel

from tests.params import HOT_GYM_MODEL_PARAMS

# rdse = random_distributed_scalar.RandomDistributedScalarEncoder()
# scalar_encoder = ScalarEncoder()
"""
Create a bunch of models and compare their time performance,
output some nice graphs.

Basically do the following:
    1. Setup an encoder and model with both nupic and htm-cl
    2. Run and compare their running time
    3. Repeat for increasing encoder size, column size, classifier steps

"""

# read all the data in


converters = {
    0: lambda s: datetime.strptime(s, '%m/%d/%y %H:%M'),
    1: lambda s: float(s)
}
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
print(data_dir)
row_count = 1000
hot_gym_data = np.genfromtxt(os.path.join(data_dir, 'rec-center-hourly.csv'),
                             dtype=[datetime, float],
                             skip_header=3,
                             delimiter=',',
                             names=['timestamp_timeOfDay', 'consumption'],
                             converters=converters)[:row_count]

params = []
for i in [2048, 4096, 8192, 16384, 32768, 65536]:
    cpy = deepcopy(HOT_GYM_MODEL_PARAMS)
    cpy['modelParams']['spParams']['columnCount'] = i
    cpy['modelParams']['tpParams']['columnCount'] = i
    params.append({
        'modelParams': cpy,
        'cols': i
    })
times = defaultdict(list)


for i in params:
    repeats = 1


    @timeit_repeat(repeats)
    def run_nupic_model():
        nu_model = NUModel(i['modelParams'])
        prog_bar = pyprind.ProgBar(row_count)
        for row in hot_gym_data:
            nu_model.run(row)
            prog_bar.update()


    @timeit_repeat(repeats)
    def run_htm_cl_model():
        cl_model = CLModel(i['modelParams'])
        prog_bar = pyprind.ProgBar(row_count)
        for row in hot_gym_data:
            cl_model.run(row)
            prog_bar.update()


    times['nupic'].append({'c': i['cols'], 't': run_nupic_model()})
    times['htmcl'].append({'c': i['cols'], 't': run_htm_cl_model()})

font = {'size': 30}
import matplotlib
import matplotlib.pyplot as plt
from pluck import pluck

matplotlib.rc('font', **font)

nupic_x = pluck(times['nupic']['c'])
nupic_y = pluck(times['nupic']['t'])

cl_x = pluck(times['htmcl']['c'])
cl_y = pluck(times['htmcl']['t'])

fig, ax = plt.subplots()
fig.plot(nupic_x, nupic_y, 'r-', label='Nupic')
fig.plot(cl_x, cl_y, 'b*', label='Nupic')
plt.legend(prop={'size': 23})
fig.subplots_adjust(bottom=0.125)

plt.grid()
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='black', linestyle='dotted')
plt.title("HTM-CL vs. Nupic for Hot Gym Dataset", y=1.03)
plt.ylabel("Running Time (s)")
plt.xlabel("Model Size (TM columns)")
plt.xticks(rotation='vertical')
