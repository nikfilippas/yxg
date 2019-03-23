import sys
import numpy as np
import pyccl as ccl
from analysis.params import ParamRun
from likelihood.like import Likelihood
from model.data import DataManager
from model.theory import get_theory

try:
    fname_params=sys.argv[1]
except:
    raise ValueError("Must provide param file name as command-line argument")

p=ParamRun(fname_params)

#Cosmology (Planck 2018)
cosmo = p.get_cosmo()

for v in p.get('data_vectors'):
    d=DataManager(p,v,cosmo)
    th=lambda pars : get_theory(p,d,cosmo,**pars)
    l=Likelihood(p.get('params'),d.data_vector,d.covar,th)
    print(th(l.build_kwargs(l.p0)))
