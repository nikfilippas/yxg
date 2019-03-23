import sys
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood

try:
    fname_params=sys.argv[1]
except:
    raise ValueError("Must provide param file name as command-line argument")

p=ParamRun(fname_params)

l=Likelihood(p.get('params'),np.ones(2),np.identity(2),gt)
