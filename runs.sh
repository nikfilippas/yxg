#!/bin/bash

for jk in {0..460}
do
    addqueue -q cmb -m 0.5 -n 1 /usr/bin/python3 bf_jk.py params_dam_wnarrow.yml ${jk}
done
