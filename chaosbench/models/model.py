import os
import torch

torch.autograd.set_detect_anomaly(True)
# from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import datetime

import numpy as np

from chaosbench.models import mlp, cnn, ae, vit, gnn, climax, sh, vgae, invgnn, egnn
from chaosbench import dataset_new, config, utils, criterion

class S2SBenchmarkModel(pl.LightningModule):

    def __init__(
        self, 
        model_args,
        data_args,
        
    ):
        super(S2SBenchmarkModel, self).__init__()
        self.save_hyperparameters()
        self.model_args = model_args
        self.data_args = data_args
        
        # Initialize model
        # ocean_vars = self.data_args.get('ocean_vars', [])
        input_size = self.model_args['input_size'] 
        output_size = self.model_args['output_size'] 
        
        if 'mlp' in self.model_args['model_name']:
            self.model = mlp.MLP(input_size = input_size,
                                 hidden_sizes = self.model_args['hidden_sizes'], 
                                 output_size = output_size)
            
        elif 'unet' in self.model_args['model_name']:
            self.model = cnn.UNet(input_size = input_size,
                                   output_size = output_size)
            
        elif 'resnet' in self.model_args['model_name']:
            self.model = cnn.ResNet(input_size = input_size,
                                   output_size = output_size)
            
        elif 'vae' in self.model_args['model_name']:
            self.model = ae.VAE(input_size = input_size,
                                 output_size = output_size,
                                 latent_size = self.model_args['latent_size'])
            
        elif 'ed' in self.model_args['model_name']:
            self.model = ae.EncoderDecoder(input_size = input_size,
                                           output_size = output_size)
            
        elif 'vit' in self.model_args['model_name']:
            self.model = vit.ViT(input_size = input_size)
        elif 'climax' in self.model_args['model_name']:
            self.model = climax.ClimaX()
        elif 'sh' in self.model_args['model_name']:
            self.model = sh.SHFormer(input_size = input_size)
            
        
        ##################################
        # INITIALIZE YOUR OWN MODEL HERE #
        ##################################
        
        self.loss = self.init_loss_fn()
        self.val_loss = criterion.MSE()
            
    def init_loss_fn(self):
        loss = criterion.MSE()
        return loss
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        timestamp, x, y = batch # x: [batch, input_size, height, width] y: [batch, step, input_size, height, width]
        x = F.pad(x, (0, 0, 0, 3), "constant", 0) 
        y = F.pad(y, (0, 0, 0, 3), "constant", 0) 
        # print(x.shape)

        n_steps = y.size(1)
        loss = 0
        preds = self(x)
        
        loss=self.loss(preds,y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        timestamp, x, y = batch # x: [batch, input_size, height, width] y: [batch, step, input_size, height, width]
        x = F.pad(x, (0, 0, 0, 3), "constant", 0) 

        n_steps = y.size(1)
        loss = 0
        preds = self(x)
        preds = preds[:, :, :, :121, :]
        
        loss=self.loss(preds,y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        timestamp, x, y = batch
        x = F.pad(x, (0, 0, 0, 3), "constant", 0) 
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0
        preds = self(x)
        preds = preds[:, :, :, :121, :]              
        
        loss=self.val_loss(preds,y)

        for i in range(63):
            loss1 = self.loss(preds[:, 0, i, :, :],y[:, 0, i, :, :])
            loss2 = self.loss(preds[:, 1, i, :, :],y[:, 1, i, :, :])
            self.log("variable" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("variable" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        timestamp, x, y = batch
        x = F.pad(x, (0, 0, 0, 3), "constant", 0) 
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0
        preds = self(x)
        preds = preds[:, :, :, :121, :]    
        
        loss=self.val_loss(preds,y)

        # for i in range(63):
        #     loss1 = self.loss(preds[:, 0, i, :, :],y[:, 0, i, :, :])
        #     loss2 = self.loss(preds[:, 1, i, :, :],y[:, 1, i, :, :])
        #     self.log("variable" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #     self.log("variable" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return preds, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_args['learning_rate'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.model_args['t_max'], eta_min=self.model_args['learning_rate'] / 10),
                'interval': 'epoch',
            }
        }

    def setup(self, stage=None):
        self.train_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                    years=self.data_args['train_years'], 
                                                   n_step=self.data_args['n_step'],
                                                   lead_time=self.data_args['lead_time'],
                                                    single_vars=self.data_args['single_vars'],
                                                    pred_single_vars=self.data_args['pred_single_vars'],
                                                    pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                   type = "image"
                                                #    ocean_vars=self.data_args['ocean_vars']
                                                  )
        self.val_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                  years=self.data_args['val_years'], 
                                                 n_step=self.data_args['n_step'],
                                                 lead_time=self.data_args['lead_time'],
                                                 single_vars=self.data_args['single_vars'],
                                                 pred_single_vars=self.data_args['pred_single_vars'],
                                                 pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                 type = "image"
                                                #  ocean_vars=self.data_args['ocean_vars']
                                                )
        
        self.test_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                  years=self.data_args['test_years'], 
                                                 n_step=self.data_args['n_step'],
                                                 lead_time=self.data_args['lead_time'],
                                                 single_vars=self.data_args['single_vars'],
                                                 pred_single_vars=self.data_args['pred_single_vars'],
                                                 pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                 type = "image"
                                                #  ocean_vars=self.data_args['ocean_vars']
                                                )
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
    
class S2SGNNModel(pl.LightningModule):

    def __init__(
        self, 
        model_args,
        data_args,
        
    ):
        super(S2SGNNModel, self).__init__()
        self.save_hyperparameters()
        
        self.model_args = model_args
        self.data_args = data_args
        
        # Initialize model
        # ocean_vars = self.data_args.get('ocean_vars', [])
        input_size = self.model_args['input_size'] 
        output_size = self.model_args['output_size']
        self.week = self.model_args['week']
        
        if 'mlp' in self.model_args['model_name']:
            self.model = mlp.MLP(input_size = input_size,
                                 hidden_sizes = self.model_args['hidden_sizes'], 
                                 output_size = output_size)
        elif 'invgnn' in self.model_args['model_name']:
            self.model = invgnn.GNN(input_dim = input_size,
                                 hidden_nf = self.model_args['hidden_sizes'], 
                                 output_dim = output_size,
                                 pred_len = self.model_args['pred_len'])
        elif 'egnn' in self.model_args['model_name']:
            self.model = egnn.EGNN(in_node_nf = input_size - 11,
                                in_edge_nf = 1,
                                hidden_nf = self.model_args['hidden_sizes'], 
                                output_dim = output_size - 22)
        elif 'gnn' in self.model_args['model_name']:
            self.model = gnn.GNN(input_dim = input_size + 2,
                                 hidden_nf = self.model_args['hidden_sizes'], 
                                 output_dim = output_size,
                                 pred_len = self.model_args['pred_len'])
        elif 'vgae' in self.model_args['model_name']:
            self.model = vgae.VGAE(input_dim = input_size,
                                 hidden_nf = self.model_args['hidden_sizes'], 
                                 output_dim = output_size,
                                 pred_len = self.model_args['pred_len'])
            
        
        ##################################
        # INITIALIZE YOUR OWN MODEL HERE #
        ##################################
        
        self.loss = self.init_loss_fn()
            
    def init_loss_fn(self):
        loss = criterion.MSE()
        return loss
    
    def forward(self, h, u=None, v=None, radial=None, edges=None, edge_attr=None, timestamp=None):
        if 'egnn' in self.model_args['model_name']:
            return self.model(h, u, v, radial, edges, edge_attr, timestamp)
        else:
            return self.model(h, edges, edge_attr)

    def training_step(self, batch, batch_idx):
        coord, x, y, edge_index, edge_feat, radial, mask = batch.coord, batch.x, batch.y,batch.edge_index, batch.edge_feat, batch.radial, batch.mask
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0

        if 'egnn' in self.model_args['model_name']:
            u = torch.cat((x[:, 30:40], x[:, 61:62]), dim=1)
            v = torch.cat((x[:, 40:50], x[:, 62:63]), dim=1)
            x = torch.cat((x[:, :30], x[:, 50:61]), dim=1)


            speed = torch.sqrt(u**2 + v**2)
            x = torch.cat([x, speed], dim=1)

            row, col = edge_index


            preds = self(h=x, u=u, v=v, radial=radial, edges=edge_index, edge_attr=edge_feat)
        elif 'invgnn' in self.model_args['model_name']:
            u = torch.cat((x[:, 30:40], x[:, 65:66]), dim=1)
            v = torch.cat((x[:, 40:50], x[:, 66:67]), dim=1)
            x = torch.cat((x[:, :30], x[:, 50:65]), dim=1)
            row, col = edge_index

            speed = torch.sqrt(u**2 + v**2)
            dirt = torch.atan2(u, v)
            x = torch.cat([x, speed, dirt], dim=1)
            preds = self(h=x, edges=edge_index, edge_attr=edge_feat)
        elif 'vgae' in self.model_args['model_name']:
            preds, kl_divergence = self(nodes=x,edges=edge_index, edge_attr=edge_feat)
        else:
            x = torch.cat([x, coord], dim=1)
            preds = self(h=x, edges=edge_index)

        # for step_idx in range(n_steps):
        #     preds = self(x)
            
        #     # Optimize for headline variables
        #     if self.model_args['only_headline']:
        #         headline_idx = [
        #             config.PARAMS.index(headline_var.split('-')[0]) * len(config.PRESSURE_LEVELS)
        #             + config.PRESSURE_LEVELS.index(int(headline_var.split('-')[1])) for headline_var in config.HEADLINE_VARS
        #         ]
                
        #         loss += self.loss(
        #             preds.view(preds.size(0), -1, preds.size(-2), preds.size(-1))[:, headline_idx],
        #             y[:, step_idx].view(y.size(0), -1, y.size(-2), y.size(-1))[:, headline_idx]
        #         )
            
        #     # Otherwise, for all variables
        #     else:
        #         loss += self.loss(preds[:, :self.model_args['output_size']], y[:, step_idx, :self.model_args['output_size']])
            
        #     x = preds
            
        # loss = loss / n_steps
        ####################################################
        # preds = torch.masked_select(preds, mask)
        # y = torch.masked_select(y[:, :, 61], mask)
        loss = self.loss(preds, y)
        # loss = self.loss(preds, y[:, :, 61])
        # if self.week == 34:
        #     loss = self.loss(preds,y[:, 0, :])
        # else:
        #     loss = self.loss(preds,y[:, 1, :])
       
        # if 'vgae' in self.model_args['model_name']:
        #     loss += kl_divergence

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        coord, x, y, edge_index, edge_feat, radial, mask = batch.coord, batch.x, batch.y,batch.edge_index, batch.edge_feat, batch.radial, batch.mask
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0

        if 'egnn' in self.model_args['model_name']:
            u = torch.cat((x[:, 30:40], x[:, 61:62]), dim=1)
            v = torch.cat((x[:, 40:50], x[:, 62:63]), dim=1)
            x = torch.cat((x[:, :30], x[:, 50:61]), dim=1)

            row, col = edge_index


            speed = torch.sqrt(u**2 + v**2)
            x = torch.cat([x, speed], dim=1)

            preds = self(h=x, u=u, v=v, radial=radial, edges=edge_index, edge_attr=edge_feat)
        elif 'invgnn' in self.model_args['model_name']:
            u = torch.cat((x[:, 30:40], x[:, 65:66]), dim=1)
            v = torch.cat((x[:, 40:50], x[:, 66:67]), dim=1)
            x = torch.cat((x[:, :30], x[:, 50:65]), dim=1)
            row, col = edge_index

            speed = torch.sqrt(u**2 + v**2)
            dirt = torch.atan2(u, v)
            x = torch.cat([x, speed, dirt], dim=1)
            preds = self(h=x, edges=edge_index, edge_attr=edge_feat)
        elif 'vgae' in self.model_args['model_name']:
            preds, kl_divergence = self(nodes=x,edges=edge_index, edge_attr=edge_feat)
        else:
            x = torch.cat([x, coord], dim=1)
            preds = self(h=x,edges=edge_index)
        
        # for step_idx in range(n_steps):
        #     preds = self(x)
            
        #     # Optimize for headline variables
        #     if self.model_args['only_headline']:
        #         headline_idx = [
        #             config.PARAMS.index(headline_var.split('-')[0]) * len(config.PRESSURE_LEVELS)
        #             + config.PRESSURE_LEVELS.index(int(headline_var.split('-')[1])) for headline_var in config.HEADLINE_VARS
        #         ]
                
        #         loss += self.loss(
        #             preds.view(preds.size(0), -1, preds.size(-2), preds.size(-1))[:, headline_idx],
        #             y[:, step_idx].view(y.size(0), -1, y.size(-2), y.size(-1))[:, headline_idx]
        #         )
                
        #     # Otherwise, for all variables
        #     else:
        #         loss += self.loss(preds[:, :self.model_args['output_size']], y[:, step_idx, :self.model_args['output_size']])
            
        #     x = preds
            
        # loss = loss / n_steps
        ####################################################

        # preds = torch.masked_select(preds, mask)
        # y = torch.masked_select(y[:, :, 61], mask)
        loss = self.loss(preds, y)
        
        # loss = self.loss(preds, y[:, :, 61])
        # if self.week == 34:
        #     loss = self.loss(preds,y[:, 0, :])
        # else:
        #     loss = self.loss(preds,y[:, 1, :])
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        coord, x, y, edge_index, edge_feat, radial = batch.coord, batch.x, batch.y,batch.edge_index, batch.edge_feat, batch.radial
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0

        if 'egnn' in self.model_args['model_name']:
            u = torch.cat((x[:, 30:40], x[:, 61:62]), dim=1)
            v = torch.cat((x[:, 40:50], x[:, 62:63]), dim=1)
            x = torch.cat((x[:, :30], x[:, 50:61]), dim=1)


            speed = torch.sqrt(u**2 + v**2)
            x = torch.cat([x, speed], dim=1)

            row, col = edge_index


            preds = self(h=x, u=u, v=v, radial=radial, edges=edge_index, edge_attr=edge_feat)
        elif 'invgnn' in self.model_args['model_name']:
            u = torch.cat((x[:, 30:40], x[:, 65:66]), dim=1)
            v = torch.cat((x[:, 40:50], x[:, 66:67]), dim=1)
            x = torch.cat((x[:, :30], x[:, 50:65]), dim=1)
            row, col = edge_index

            speed = torch.sqrt(u**2 + v**2)
            dirt = torch.atan2(u, v)
            x = torch.cat([x, speed, dirt], dim=1)

            preds = self(h=x, edges=edge_index, edge_attr=edge_feat)
        elif 'vgae' in self.model_args['model_name']:
            preds, kl_divergence = self(nodes=x,edges=edge_index, edge_attr=edge_feat)
        else:
            x = torch.cat([x, coord], dim=1)
            # return x, edge_index
            preds = self(h=x,edges=edge_index)

        loss = self.loss(preds, y)
        
        # loss = self.loss(preds, y[:, :, 61])
        return preds, y, loss
    
    def test_step(self, batch, batch_idx):
        coord, x, y, edge_index, edge_feat, radial, mask = batch.coord, batch.x, batch.y,batch.edge_index, batch.edge_feat, batch.radial, batch.mask
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0

        if 'egnn' in self.model_args['model_name']:
            u = torch.cat((x[:, 30:40], x[:, 61:62]), dim=1)
            v = torch.cat((x[:, 40:50], x[:, 62:63]), dim=1)
            x = torch.cat((x[:, :30], x[:, 50:61]), dim=1)

            speed = torch.sqrt(u**2 + v**2)
            x = torch.cat([x, speed], dim=1)

            row, col = edge_index

            preds = self(h=x, u=u, v=v, radial=radial, edges=edge_index, edge_attr=edge_feat)
        elif 'invgnn' in self.model_args['model_name']:
            u = torch.cat((x[:, 30:40], x[:, 65:66]), dim=1)
            v = torch.cat((x[:, 40:50], x[:, 66:67]), dim=1)
            x = torch.cat((x[:, :30], x[:, 50:65]), dim=1)
            speed = torch.sqrt(u**2 + v**2)
            dirt = torch.atan2(u, v)
            x = torch.cat([x, speed, dirt], dim=1)
            
            
            preds = self(h=x, edges=edge_index, edge_attr=edge_feat)
        elif 'vgae' in self.model_args['model_name']:
            preds, kl_divergence = self(nodes=x,edges=edge_index, edge_attr=edge_feat)
        else:
            x = torch.cat([x, coord], dim=1)
            preds = self(h=x,edges=edge_index)

        # preds = torch.masked_select(preds, mask)
        # y = torch.masked_select(y[:, :, 61], mask)
        loss = self.loss(preds, y)

        # loss = self.loss(preds, y[:, :, 61])

        # if self.week == 34:
        #     loss = self.loss(preds,y[:, 0, :])
        # else:
        #     loss = self.loss(preds,y[:, 1, :])
        
        # for step_idx in range(n_steps):
        #     preds = self(x)
            
        #     # Optimize for headline variables
        #     if self.model_args['only_headline']:
        #         headline_idx = [
        #             config.PARAMS.index(headline_var.split('-')[0]) * len(config.PRESSURE_LEVELS)
        #             + config.PRESSURE_LEVELS.index(int(headline_var.split('-')[1])) for headline_var in config.HEADLINE_VARS
        #         ]
                
        #         loss += self.loss(
        #             preds.view(preds.size(0), -1, preds.size(-2), preds.size(-1))[:, headline_idx],
        #             y[:, step_idx].view(y.size(0), -1, y.size(-2), y.size(-1))[:, headline_idx]
        #         )
                
        #     # Otherwise, for all variables
        #     else:
        #         loss += self.loss(preds[:, :self.model_args['output_size']], y[:, step_idx, :self.model_args['output_size']])
            
        #     x = preds
            
        # loss = loss / n_steps
        ####################################################
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # self.log("variable61", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for i in range(63):
            loss1 = self.loss(preds[:, 0, i],y[:, 0, i])
            loss2 = self.loss(preds[:, 1, i],y[:, 1, i])
            self.log("variable" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("variable" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # for i in range(69):
        #     if self.week == 34:
        #         loss = self.loss(preds[:, i],y[:, 0, i])
        #         self.log("variable" + str(i) + " Week34", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #     else:
        #         loss = self.loss(preds[:, i],y[:, 0, i])
        #         self.log("variable" + str(i) + " Week56", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
            

        # for i in range(10):
        #     loss1 = self.loss(preds[:, 0, i],y[:, 0, i])
        #     loss2 = self.loss(preds[:, 1, i],y[:, 1, i])

        #     self.log("z" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #     self.log("z" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # for i in range(10, 20):
        #     loss1 = self.loss(preds[:, 0, i],y[:, 0, i])
        #     loss2 = self.loss(preds[:, 1, i],y[:, 1, i])

        #     self.log("q" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #     self.log("q" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # for i in range(20, 30):
        #     loss1 = self.loss(preds[:, 0, i],y[:, 0, i])
        #     loss2 = self.loss(preds[:, 1, i],y[:, 1, i])

        #     self.log("t" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #     self.log("t" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # for i in range(40, 49):
        #     loss1 = self.loss(preds[:, 0, i],y[:, 0, i])
        #     loss2 = self.loss(preds[:, 1, i],y[:, 1, i])

        #     self.log("single" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #     self.log("single" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_args['learning_rate'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.model_args['t_max'], eta_min=self.model_args['learning_rate'] / 10),
                'interval': 'epoch',
            }
        }

    def setup(self, stage=None):
        self.train_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                    years=self.data_args['train_years'], 
                                                    n_step=self.data_args['n_step'],
                                                    lead_time=self.data_args['lead_time'],
                                                    kernel_size=self.data_args['kernel_size'],
                                                    single_vars=self.data_args['single_vars'],
                                                    pred_single_vars=self.data_args['pred_single_vars'],
                                                    pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                    type = "graph",
                                                #    ocean_vars=self.data_args['ocean_vars']
                                                  )
        self.val_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                    years=self.data_args['val_years'], 
                                                    n_step=self.data_args['n_step'],
                                                    lead_time=self.data_args['lead_time'],
                                                    kernel_size=self.data_args['kernel_size'],
                                                    single_vars=self.data_args['single_vars'],
                                                    pred_single_vars=self.data_args['pred_single_vars'],
                                                    pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                    type = "graph",
                                                #  ocean_vars=self.data_args['ocean_vars']
                                                )
        self.test_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                    years=self.data_args['test_years'], 
                                                    n_step=self.data_args['n_step'],
                                                    lead_time=self.data_args['lead_time'],
                                                    kernel_size=self.data_args['kernel_size'],
                                                    single_vars=self.data_args['single_vars'],
                                                    pred_single_vars=self.data_args['pred_single_vars'],
                                                    pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                    type = "graph",
                                                #  ocean_vars=self.data_args['ocean_vars']
                                                )
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
