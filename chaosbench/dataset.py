import torch
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
import glob
import xarray as xr
import numpy as np
from datetime import datetime
import re

from chaosbench import config

class S2SObsDataset(Dataset):
    """
    Dataset object to handle input reanalysis.
    
    Params:
        years <List[int]>      : list of years to load and process,
        n_step <int>           : number of contiguous timesteps included in the data (default: 1)
        lead_time <int>        : delta_t ahead in time, useful for direct prediction (default: 1)
        land_vars <List[str]>  : list of land variables to include (default: empty)`
        ocean_vars <List[str]> : list of sea/ice variables to include (default: empty)`
        is_normalized <bool>   : flag to indicate whether we should perform normalization or not (default: True)
    """
    
    def __init__(
        self, 
        years: List[int],
        n_step: int = 1,
        lead_time: int = 1,
        land_vars: List[str] = [],
        # ocean_vars: List[str] = [],
        is_normalized: bool = True
    ) -> None:
        
        self.data_dir = [
            Path(config.DATA_DIR) / 'pressure_level',
            Path(config.DATA_DIR) / 'single_level',
            # Path(config.DATA_DIR) / 'oras5'
        ]
        
        self.normalization_file = [
            Path(config.DATA_DIR) / 'climatology' / 'climatology_pressure_level.zarr',
            Path(config.DATA_DIR) / 'climatology' / 'climatology_single_level_merge.zarr',
            # Path(config.DATA_DIR) / 'climatology' / 'climatology_oras5.zarr'
        ]
        
        self.years = [str(year) for year in years]
        self.n_step = n_step
        self.lead_time = lead_time
        self.land_vars = land_vars
        # self.ocean_vars = ocean_vars
        self.is_normalized = is_normalized
        
        # Subset files that match with patterns (eg. years specified)
        # pressure_level_files, single_level_merge_files, oras5_files = list(), list(), list()
        pressure_level_files, single_level_merge_files = list(), list()
        for year in self.years:
            pattern = rf'.*{year}\d{{4}}\.zarr$'
            
            curr_files = [
                list(self.data_dir[0].glob(f'*{year}*.zarr')),
                list(self.data_dir[1].glob(f'*{year}*.zarr')),
                # list(self.data_dir[2].glob(f'*{year}*.zarr'))
            ]
            
            pressure_level_files.extend([f for f in curr_files[0] if re.match(pattern, str(f.name))])
            single_level_merge_files.extend([f for f in curr_files[1] if re.match(pattern, str(f.name))])
            # oras5_files.extend([f for f in curr_files[2] if re.match(pattern, str(f.name))])
        
        # pressure_level_files.sort(); single_level_merge_files.sort(); oras5_files.sort()
        pressure_level_files.sort(); single_level_merge_files.sort()
        self.file_paths = [pressure_level_files, single_level_merge_files]
        
        # Subsetting
        single_level_merge_idx = [idx for idx, param in enumerate(config.single_level_merge_PARAMS) if param in self.land_vars]
        # oras5_idx = [idx for idx, param in enumerate(config.ORAS5_PARAMS) if param in self.ocean_vars]
        
        # Retrieve climatology (i.e., mean and sigma) to normalize
        self.mean_pressure_level = xr.open_dataset(self.normalization_file[0], engine='zarr')['mean'].values[:, np.newaxis, np.newaxis]
        self.mean_single_level_merge = xr.open_dataset(self.normalization_file[1], engine='zarr')['mean'].values[single_level_merge_idx, np.newaxis, np.newaxis]
        # self.mean_oras5 = xr.open_dataset(self.normalization_file[2], engine='zarr')['mean'].values[oras5_idx, np.newaxis, np.newaxis]
        
        self.sigma_pressure_level = xr.open_dataset(self.normalization_file[0], engine='zarr')['sigma'].values[:, np.newaxis, np.newaxis]
        self.sigma_single_level_merge = xr.open_dataset(self.normalization_file[1], engine='zarr')['sigma'].values[single_level_merge_idx, np.newaxis, np.newaxis]
        # self.sigma_oras5 = xr.open_dataset(self.normalization_file[2], engine='zarr')['sigma'].values[oras5_idx, np.newaxis, np.newaxis]
        

    def __len__(self):
        data_length = len(self.file_paths[0]) - self.n_step - self.lead_time
        return data_length

    def __getitem__(self, idx):
        step_indices = [idx] + [target_idx for target_idx in range(idx + self.lead_time, idx + self.lead_time + self.n_step)]
        
        # pressure_level_data, single_level_merge_data, oras5_data = list(), list(), list()
        pressure_level_data, single_level_merge_data = list(), list()
        
        for step_idx in step_indices:
            
            # # Process pressure_level
            pressure_level_data.append(xr.open_dataset(self.file_paths[0][step_idx], engine='zarr')[config.PARAMS].to_array().values)
            
            # Process single_level_merge
            if len(self.land_vars) > 0:
                single_level_merge_data.append(xr.open_dataset(self.file_paths[1][step_idx], engine='zarr')[self.land_vars].to_array().values)
            
            # Process oras5
            # if len(self.ocean_vars) > 0:
            #     oras5_data.append(xr.open_dataset(self.file_paths[2][step_idx], engine='zarr')[self.ocean_vars].to_array().values)
        
        # Permutation / reshaping
        pressure_level_data, single_level_merge_data = np.array(pressure_level_data), np.array(single_level_merge_data)
        pressure_level_data = pressure_level_data.reshape(pressure_level_data.shape[0], -1, pressure_level_data.shape[-2], pressure_level_data.shape[-1]) # Merge (param, level) dims
        
        # Normalize
        if self.is_normalized:
            pressure_level_data = (pressure_level_data - self.mean_pressure_level[np.newaxis, :, :, :]) / self.sigma_pressure_level[np.newaxis, :, :, :]
            single_level_merge_data = (single_level_merge_data - self.mean_single_level_merge[np.newaxis, :, :, :]) / self.sigma_single_level_merge[np.newaxis, :, :, :]
            # oras5_data = (oras5_data - self.mean_oras5[np.newaxis, :, :, :]) / self.sigma_oras5[np.newaxis, :, :, :]
        
        # Concatenate along parameter dimension, only if they are specified (i.e., non-empty)
        data = [t for t in [torch.tensor(pressure_level_data), torch.tensor(single_level_merge_data)] if t.nelement() > 0]
        data = torch.cat(data, dim=1)
        
        x, y = data[0].float(), data[1:].float()
        timestamp = xr.open_dataset(self.file_paths[0][idx], engine='zarr').time.values.item()

        return timestamp, x, y
    
    
class S2SEvalDataset(Dataset):
    """
    Dataset object to load evaluation benchmarks.
    
    Params:
        s2s_name <str>        : center name where evaluation is going to be performed
        years <List[int]>     : list of years to load and process
        is_ensemble <bool>    : indicate whether to use control or perturbed ensemble forecasts
        is_normalized <bool>  : flag to indicate whether we should perform normalization or not (default: True)
    """
    
    def __init__(
        self, 
        s2s_name: str,
        years: List[int],
        is_ensemble = False,
        is_normalized = True
    ) -> None:
        
        assert s2s_name in list(config.S2S_CENTERS.keys())
        
        self.s2s_name = s2s_name
        self.data_dir = Path(config.DATA_DIR) / f'{self.s2s_name}_ensemble' if is_ensemble else Path(config.DATA_DIR) / self.s2s_name 
        self.normalization_file = Path(config.DATA_DIR) / 'climatology' / f'climatology_{self.s2s_name}.zarr'
        
        # Check if years specified are within valid bounds
        self.years = years
        self.years = [str(year) for year in self.years]
        assert set(self.years).issubset(set(config.YEARS))
        
        self.is_ensemble = is_ensemble
        self.is_normalized = is_normalized

        # Subset files that match with patterns (eg. years specified)
        file_paths = list()
        
        for year in self.years:
            pattern = rf'.*{year}\d{{4}}\.zarr$'
            curr_files = list(self.data_dir.glob(f'*{year}*.zarr'))
            file_paths.extend(
                [f for f in curr_files if re.match(pattern, str(f.name))]
            )
            
        # Subset files that match with patterns (eg. years specified)
        self.file_paths = file_paths
        self.file_paths.sort()
        
        # Retrieve climatology to normalize
        self.normalization = xr.open_dataset(self.normalization_file, engine='zarr')
        self.normalization_mean = self.normalization['mean'].values[:, np.newaxis, np.newaxis]
        self.normalization_sigma = self.normalization['sigma'].values[:, np.newaxis, np.newaxis]
        

    def __len__(self):
        return (len(self.file_paths) - config.N_STEPS)

    def __getitem__(self, idx):
        data = xr.open_dataset(self.file_paths[idx], engine='zarr')
        data = data[config.PARAMS].to_array().values
        
        if self.is_ensemble:
            data = torch.tensor(data).permute((2, 1, 0, 3, 4, 5)) # Shape: (step, ensem, param, level, lat, lon)
            data = data.reshape(data.shape[0], data.shape[1], -1, data.shape[-2], data.shape[-1]) # Shape: (step, ensem, param*level, lat, lon)
            
            if self.is_normalized:
                data = (data - self.normalization_mean[np.newaxis, np.newaxis, :, :, :]) / self.normalization_sigma[np.newaxis, np.newaxis, :, :, :] # Normalize

            x, y = data[0], data[1:]
            timestamp = xr.open_dataset(self.file_paths[idx], engine='zarr').time.values.item()
        
        else:
            data = torch.tensor(data).permute((1, 0, 2, 3, 4)) # Shape: (step, param, level, lat, lon)
            data = data.reshape(data.shape[0], -1, data.shape[-2], data.shape[-1]) # Shape: (step, param*level, lat, lon)
            
            if self.is_normalized:
                data = (data - self.normalization_mean[np.newaxis, :, :, :]) / self.normalization_sigma[np.newaxis, :, :, :] # Normalize

            x, y = data[0], data[1:]
            timestamp = xr.open_dataset(self.file_paths[idx], engine='zarr').time.values.item()
        
        return timestamp, x, y
