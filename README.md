[![DOI](https://zenodo.org/badge/930178705.svg)](https://doi.org/10.5281/zenodo.15516467)
# create environment
```
conda create -n eimp python==3.9
conda activate eimp
pip install -r requirements.txt
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```
# Download and Process Data
The traning data is downloaded from [WeatherBench2](https://console.cloud.google.com/storage/browser/weatherbench2;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)
```
python data_download/training_data/step1_pressure_level_download.py

python data_download/training_data/step2_single_level_download.py

python data_download/training_data/step3_compute_climatology.py --dataset_name pressure_level_1.5

python data_download/training_data/step3_compute_climatology.py --dataset_name single_level_1.5
```

# Train EIMP
```
python train.py
```
## Acknowledgement
We use the code from the repository [ChaosBench](https://github.com/leap-stc/ChaosBench)