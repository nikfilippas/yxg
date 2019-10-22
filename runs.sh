#!/bin/bash

for jk in {0..460}
do
    python bf_jk.py params_wnarrow.yml ${jk}
done
