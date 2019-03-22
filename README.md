# YxG

## Running the pipeline
1. Unpack and download all necessary data by running the bash script `dwl_data.sh`. This will unpack `data.tar.gz`, download all necessary Planck maps and compute the SZ masks using `mk_mask_sz.py`.
2. Compute all power spectra and covariance matrices running `python pipeline.py params.yaml`.

## Source code
- `dwl_data.sh` downloads all the data needed for the analysis.
- All the data analysis modules can be found in `analysis`.
- The theory prediction modules can be found in `model`.
- `pipeline.py` contains the power spectrum measurement pipeline.

## Theory notes
Theory notes can be found in `notes`
