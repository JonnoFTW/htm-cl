HTM-CL
======

Heirarchical Temporal Memory using OpenCL to speed things up. Ideally it would eventually
be a drop in replacement for the python or c++ implementations, which means the components implemented need
to implement the same interface and pass the same tests as the nupic versions. The end goal would be that you would only
need to install this package and specify in your model params:

```
'spParams': {
    ...
    'temporalImp':'cl'
    ...
},
'tpParams': {
    ...
    'temporalImp':'cl'
    ...
},
'clParams': {
    ...
    'implementation':'cl'
    ...
}
```

Parallelisation
===============

Opportunities for parallelising the various parts of the numerous algorithms involved in HTM are:

Encoding
--------

There are usually very few model  inputs (< 10 typically), so parallising this would likely not
provide performance improvements due to the
extra overhead.

Spatial Pooler
--------------
Since each column has many synapses, we can easily parallelise column level operations:

* Calculating overlap: each column is a single work group, overlap boosting is also done here
* Updating permanences after the set of active columns is decided
* Updating boost factors


Temporal memory
---------------

* Columns can be processed in parallel during inference and learning

CLA Classifier
--------------
Since every time step we request a prediction for requires a new set of table, each step can be done
in parallel

* Updating the moving average for each corresponding on-bit of the input can also be parallelised

SDR Classifier
--------------

I'm not familiar with this classifier, but I understand it is intended to replace the CLA classifier