[![Build Status](https://travis-ci.com/jmsull/gzpt.svg?token=qyXyxSxrxC9pHePgsUAV&branch=master)](https://travis-ci.com/jmsull/gzpt)
[![image](http://img.shields.io/pypi/v/gzpt.svg)](https://pypi.python.org/pypi/gzpt/)

# gzpt
Hybrid Perturbation Theory + Halo Model 2-point statistics

gzpt provides a simple implementation of the analytic expressions used in [Sullivan, Seljak \& Singh (2021)](https://arxiv.org/pdf/2104.10676.pdf)

## Installation
Using pip:
```
pip install gzpt
```

## Dependencies:
 - numpy, scipy
 - pyfftw


## Simple Example


```python
from gzpt import hzpt,matter,tracers
import numpy as np

# Provide a linear theory power spectrum at some z and instantiate hzpt model
klin,plin = np.loadtxt('./tests/test_plin_cc_z0.55.txt',unpack=True)
model = hzpt(klin,plin)

#set up correlator objects for matter, tracer cross and auto, nmax is inferred from size of parameters
A0,R,R1h,R1sq,R12 = 350.,26.,5.,20.,2.
nbar,b1,Rexc = 1e-3, 2., 2.
mm = matter.Correlator([A0,R,R1h,R1sq,R12],model)
tm = tracers.CrossCorrelator([b1,A0,R,R1h,R1sq,R12],model)
tt = tracers.AutoCorrelator([nbar,b1,A0,R,R1h],model,params_exc=[Rexc]) #use one exclusion parameter


kk = np.logspace(-3,np.log10(2),1000)
rr = np.logspace(0,2,1000)

#get some matter correlators
Pmm = mm.Power()(kk)
Ximm,Ximm_grad = mm.Xi(wantGrad=True)(rr) #get the grad if you want it

```
A more involved example is provided in docs/first_example.ipynb

The Zel'dovich correlator code is based on code of Stephen Chen (https://github.com/sfschen/velocileptors) and Chirag Modi (https://github.com/modichirag/CLEFT), which is in turn built upon Yin Li's mcfit package (https://github.com/eelregit/mcfit) and Martin White's CLEFT code (https://github.com/martinjameswhite/CLEFT_GSM).

Disclaimer that the organization of the code could be greatly improved!
