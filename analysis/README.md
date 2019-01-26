# Running the pipeline

1. Unpack and download all necessary data by running the bash script `dwl_data.sh`. This will unpack `data.tar.gz`, download all necessary Planck maps and compute the SZ masks using `mk_mask_sz.fits`.
2. Compute power spectra using `analysis.py`. To view the different pipeline options run `python analysis.py -h`.
