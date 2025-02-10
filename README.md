# create environment
```
conda create -n eimp python==3.9
conda activate eimp
pip install -r requirements.txt
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```
# Download and Process Data
The traning data is downloaded from [WeatherBench2](https://console.cloud.google.com/storage/browser/weatherbench2;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)


# Train EIMP
```
python train.py
```
