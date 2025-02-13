import torch
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
import glob
import xarray as xr
import numpy as np
from datetime import datetime
import re
from tqdm import tqdm
from chaosbench import config
from torch_geometric.data import Data
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    distance = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # R = 6371.0
    # distance = R * c

    return distance

class S2SDataset(Dataset):
    """
    Dataset object to handle input reanalysis.
    
    Params:
        years <List[int]>      : list of years to load and process,
        n_step <int>           : number of contiguous timesteps included in the data (default: 1)
        lead_time <int>        : delta_t ahead in time, useful for direct prediction (default: 1)
        single_vars <List[str]>  : list of land variables to include (default: empty)`
        ocean_vars <List[str]> : list of sea/ice variables to include (default: empty)`
        is_normalized <bool>   : flag to indicate whether we should perform normalization or not (default: True)
    """
    
    def __init__(
        self, 
        data_dir: str,
        years: List[int],
        n_step: int = 1,
        lead_time: int = 1,
        kernel_size: int = 4,
        single_vars: List[str] = [],
        pred_single_vars: List[str] = [],
        pred_pressure_vars: List[str] = [],
        is_normalized: bool = True,
        type: str = "graph"
    ) -> None:
        self.type=type
        if self.type == "graph":
            self.num_nodes = 121*240
            self.edge_index, self.edge_feat, self.radial = self._create_edges(121, 240, kernel_size)
            self.coord = self._create_coord()
        self.data_dir = [
            Path(data_dir) / 'pressure_level_1.5',
            Path(data_dir) / 'single_level_1.5',
            # Path(config.DATA_DIR) / 'oras5'
        ]
        self.normalization_file = [
            Path(data_dir) / 'climatology_1.5' / 'climatology_pressure_level_1.5_new.zarr',
            Path(data_dir) / 'climatology_1.5' / 'climatology_single_level_1.5_new.zarr',
            # Path(config.DATA_DIR) / 'climatology' / 'climatology_oras5.zarr'
        ]

        self.mask = torch.tensor(np.load('/home/GraphS2S/land_mask.npy')).unsqueeze(-1)
        
        self.years = [str(year) for year in years]
        self.n_step = n_step
        self.lead_time = lead_time
        self.single_vars = single_vars
        self.pred_single_vars = pred_single_vars
        self.pred_pressure_vars = pred_pressure_vars
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
        single_level_merge_idx = [idx for idx, param in enumerate(config.SINGLE_LEVEL_PARAMS) if param in self.single_vars]
        # oras5_idx = [idx for idx, param in enumerate(config.ORAS5_PARAMS) if param in self.ocean_vars]
        
        # Retrieve climatology (i.e., mean and sigma) to normalize
        self.mean_pressure_level = xr.open_dataset(self.normalization_file[0], engine='zarr')['mean'].values[:, np.newaxis, np.newaxis]
        self.mean_single_level_merge = xr.open_dataset(self.normalization_file[1], engine='zarr')['mean'].sel(param=self.single_vars).values[:, np.newaxis, np.newaxis]
        self.mean_pressure_level_pred = xr.open_dataset(self.normalization_file[0], engine='zarr')['mean'].sel(param=[f"{param}-{level}" for param in self.pred_pressure_vars for level in config.PRESSURE_LEVELS]).values[:, np.newaxis, np.newaxis]
        self.mean_single_level_merge_pred = xr.open_dataset(self.normalization_file[1], engine='zarr')['mean'].sel(param=self.pred_single_vars).values[:, np.newaxis, np.newaxis]
        # self.mean_oras5 = xr.open_dataset(self.normalization_file[2], engine='zarr')['mean'].values[oras5_idx, np.newaxis, np.newaxis]
        
        self.sigma_pressure_level = xr.open_dataset(self.normalization_file[0], engine='zarr')['sigma'].values[:, np.newaxis, np.newaxis]
        self.sigma_single_level_merge = xr.open_dataset(self.normalization_file[1], engine='zarr')['sigma'].sel(param=self.single_vars).values[:, np.newaxis, np.newaxis]
        self.sigma_pressure_level_pred = xr.open_dataset(self.normalization_file[0], engine='zarr')['sigma'].sel(param=[f"{param}-{level}" for param in self.pred_pressure_vars for level in config.PRESSURE_LEVELS]).values[:, np.newaxis, np.newaxis]
        self.sigma_single_level_merge_pred = xr.open_dataset(self.normalization_file[1], engine='zarr')['sigma'].sel(param=self.pred_single_vars).values[:, np.newaxis, np.newaxis]
        # self.sigma_oras5 = xr.open_dataset(self.normalization_file[2], engine='zarr')['sigma'].values[oras5_idx, np.newaxis, np.newaxis]
        

    def __len__(self):
        data_length = len(self.file_paths[0]) - self.n_step - self.lead_time
        return data_length

    def __getitem__(self, idx):
        pred_indices =  [target_idx for target_idx in range(idx + self.lead_time, idx + self.lead_time + self.n_step)]
        # [idx] 
        # pressure_level_data, single_level_merge_data, oras5_data = list(), list(), list()
        pressure_level_data, single_level_merge_data = list(), list()
        pressure_level_data_pred, single_level_merge_data_pred = list(), list()

        pressure_level_data.append(xr.open_dataset(self.file_paths[0][idx], engine='zarr')[config.ERA5_PRESSURE_LIST].to_array().values)
        
        # Process single_level_merge
        if len(self.single_vars) > 0:
            single_level_merge_data.append(xr.open_dataset(self.file_paths[1][idx], engine='zarr')[self.single_vars].to_array().values)

        for step_idx in pred_indices:
            
            # # Process pressure_level
            pressure_level_data_pred.append(xr.open_dataset(self.file_paths[0][step_idx], engine='zarr')[self.pred_pressure_vars].to_array().values)
            
            # Process single_level_merge
            if len(self.single_vars) > 0:
                single_level_merge_data_pred.append(xr.open_dataset(self.file_paths[1][step_idx], engine='zarr')[self.pred_single_vars].to_array().values)
            
            # Process oras5
            # if len(self.ocean_vars) > 0:
            #     oras5_data.append(xr.open_dataset(self.file_paths[2][step_idx], engine='zarr')[self.ocean_vars].to_array().values)
        
        # Permutation / reshaping
        pressure_level_data, single_level_merge_data = np.array(pressure_level_data), np.array(single_level_merge_data)
        pressure_level_data = pressure_level_data.reshape(pressure_level_data.shape[0], -1, pressure_level_data.shape[-2], pressure_level_data.shape[-1]) # Merge (param, level) dims

        pressure_level_data_pred, single_level_merge_data_pred = np.array(pressure_level_data_pred), np.array(single_level_merge_data_pred)
        pressure_level_data_pred = pressure_level_data_pred.reshape(pressure_level_data_pred.shape[0], -1, pressure_level_data_pred.shape[-2], pressure_level_data_pred.shape[-1]) # Merge (param, level) dims
        # Normalize
        if self.is_normalized:
            pressure_level_data = (pressure_level_data - self.mean_pressure_level[np.newaxis, :, :, :]) / self.sigma_pressure_level[np.newaxis, :, :, :]
            single_level_merge_data = (single_level_merge_data - self.mean_single_level_merge[np.newaxis, :, :, :]) / self.sigma_single_level_merge[np.newaxis, :, :, :]
            pressure_level_data_pred = (pressure_level_data_pred - self.mean_pressure_level_pred[np.newaxis, :, :, :]) / self.sigma_pressure_level_pred[np.newaxis, :, :, :]
            single_level_merge_data_pred = (single_level_merge_data_pred - self.mean_single_level_merge_pred[np.newaxis, :, :, :]) / self.sigma_single_level_merge_pred[np.newaxis, :, :, :]
        
        # Concatenate along parameter dimension, only if they are specified (i.e., non-empty)
        input_data = [t for t in [torch.tensor(pressure_level_data), torch.tensor(single_level_merge_data)] if t.nelement() > 0]
        input_data = torch.cat(input_data, dim=1)

        output_data = [t for t in [torch.tensor(pressure_level_data_pred), torch.tensor(single_level_merge_data_pred)] if t.nelement() > 0]
        output_data = torch.cat(output_data, dim=1)

        timestamp = xr.open_dataset(self.file_paths[0][idx], engine='zarr').time.values.item()
        timestamp = datetime.fromtimestamp(timestamp / 1000000000)
        timestamp = timestamp.day
        timestamp = torch.tensor(np.array([np.sin(2 * np.pi *  timestamp), np.cos(2 * np.pi *  timestamp), np.sin(2 * np.pi *  timestamp / 365), np.cos(2 * np.pi *  timestamp / 365)]), dtype=torch.float)

        # x, y = input_data[0].float(), output_data[:self.n_step].float()
        x, y = input_data[0].float(), torch.stack([torch.mean(output_data[0:14].float(),dim=0), torch.mean(output_data[14:28].float(),dim=0)],dim=0)

        # lat = np.arange(90, -91, -1.5)
        # lon = np.arange(0, 360, 1.5) - 180
        # lonlat = []
        # for la in lat:
        #     for lo in lon:
        #         lonlat.append([lo, la])
        # lonlat = torch.tensor(lonlat, dtype=torch.float32)
        # embedded_lonlat = self.sh(lonlat).reshape(self.L * self.L, 121, 240)
        # x = torch.cat([x, embedded_lonlat], dim=0)
        # print(output_data[14:28, :, :, :].shape)
        # y = torch.cat([output_data[:28].float(), y], dim=0)
        if self.type=="graph":
            x = x.permute(1, 2, 0).reshape(self.num_nodes, -1)#(num_nodes,feature)
            y = y.permute(2, 3, 0, 1).reshape(self.num_nodes,y.shape[0],y.shape[1] )#(num_nodes,week,feature)
            dataset = Data(x=x, edge_index=self.edge_index, edge_feat=self.edge_feat, radial=self.radial, y=y, coord=self.coord, mask=self.mask)
            return dataset
        elif self.type=="image":
            return timestamp, x, y
        else:
            raise ValueError(f"Unsupported type: {self.type}. Expected 'graph' or 'image'.")
    
    def _create_coord(self):
        coord = []
        for lat in tqdm(range(121)):
            for lon in range(240):
                coord.append((lat / 121, lon / 240))

        coord = torch.tensor(coord, dtype=torch.float)
        return coord


    def _create_edges(self, latitude, longitude, kernel_size):
        print("--------creating edge--------")
        edge = []
        edge_feat = []
        radial = []
        for lat in tqdm(range(latitude)):
            for lon in range(longitude):

                min_lat = max(0, lat - kernel_size)
                max_lat = min(latitude - 1, lat + kernel_size)

                min_lon = lon - kernel_size
                max_lon = lon + kernel_size

                for la in range(min_lat, max_lat + 1):
                    for lo in range(min_lon, max_lon + 1):
                        # if la != lat or lo != lon:
                        edge.append((lat, lon, la, lo % longitude))
                        edge_feat.append([haversine_distance(lat, lon, la, lo % longitude)])
                        # edge_feat.append([haversine_distance(lat, lon, la, lo % longitude)])
                        # edge_feat.append([lat, lon, la, lo % longitude])
                        radial.append([lat - la, lon - lo % longitude])

        edge_index = [(e[0] * longitude + e[1], e[2] * longitude + e[3]) for e in edge]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        edge_feat = torch.tensor(edge_feat, dtype=torch.float)
        radial = torch.tensor(radial, dtype=torch.long)
        return edge_index, edge_feat, radial


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
