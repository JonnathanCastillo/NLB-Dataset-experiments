#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import os, gc
import numpy as np
import math
from icecream import ic


import wandb
from datetime import datetime

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors
from nlb_tools.evaluation import evaluate

device = torch.device('cuda:0')
wandb.login()

def make_batches(train_input, train_output, batch_size):
    
    remainder = len(train_input) % batch_size
    train_input_resized = train_input[0: len(train_input) - remainder]
    train_output_resized = train_output[0: len(train_output) - remainder]
    
    input_batch_list = torch.reshape(train_input_resized, (len(train_input_resized)//batch_size, batch_size, len(train_input_resized[0]), len(train_input_resized[0][0])))
    output_batch_list = torch.reshape(train_output_resized, (len(train_output_resized)//batch_size, batch_size, len(train_output_resized[0]), len(train_output_resized[0][0])))
    
    
        
    return (input_batch_list, output_batch_list)

#SAM did not perform as well as hoped in this case. Implemented my own, then tried this implementation from papers with code incase of unseen bugs, but no difference in performance
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

DATAPATH_DICT = {
    'mc_maze': '/home/user/Desktop/nlb-lightning/data/dandi/000128/sub-Jenkins/',
    'mc_rtt': '/home/user/Desktop/nlb-lightning/data/dandi/000129/sub-Indy/',
    'area2_bump': '/home/user/Desktop/nlb-lightning/data/dandi/000127/sub-Han/',
    'dmfc_rsg': '/home/user/Desktop/nlb-lightning/data/dandi/000130/sub-Haydn/',
    'mc_maze_large': '/home/user/Desktop/nlb-lightning/data/dandi/000138/sub-Jenkins/',
    'mc_maze_medium': '/home/user/Desktop/nlb-lightning/data/dandi/000139/sub-Jenkins/',
    'mc_maze_small': '/home/user/Desktop/nlb-lightning/data/dandi/000140/sub-Jenkins/',
}

def get_data(dataset_name, phase = 'test', bin_size = 5):
    """Function that extracts and formats data for training model"""
    dataset = NWBDataset(DATAPATH_DICT[dataset_name], skip_fields=['cursor_pos', 'eye_pos', 'cursor_vel', 'eye_vel', 'hand_pos'])
    dataset.resample(5)
    train_split = ['train', 'val'] if phase == 'test' else 'train'
    eval_split = phase
    train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=False, include_forward_pred=True)
    eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=False)
    training_input = np.concatenate([
        train_dict['train_spikes_heldin'],
        np.zeros(train_dict['train_spikes_heldin_forward'].shape),
    ], axis=1)
    training_output = np.concatenate([
        np.concatenate([
            train_dict['train_spikes_heldin'],
            train_dict['train_spikes_heldin_forward'],
        ], axis=1),
        np.concatenate([
            train_dict['train_spikes_heldout'],
            train_dict['train_spikes_heldout_forward'],
        ], axis=1),
    ], axis=2)
    eval_input = np.concatenate([
        eval_dict['eval_spikes_heldin'],
        np.zeros((
            eval_dict['eval_spikes_heldin'].shape[0],
            train_dict['train_spikes_heldin_forward'].shape[1],
            eval_dict['eval_spikes_heldin'].shape[2]
        )),
    ], axis=1)
    del dataset
    return training_input, training_output, eval_input


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, lr, model_init, dropout=0.0, input_dropout=0.0):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            hidden_dim - Hidden dimensionality to use inside the Transformer
            output_dim - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(input_dim, hidden_dim)
        )
        
        # Transformer
        self.transformer = TransformerEncoder(num_layers = num_layers,
                                              input_dim = hidden_dim,
                                              dim_feedforward = 4 * hidden_dim,
                                              num_heads = num_heads,
                                              dropout = dropout)
        
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x, mask = None):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs
        """
        
        x = self.input_net(x)
        
        x = self.transformer(x, mask = mask)
        x = self.output_net(x)
        x = self.output_norm(x) #added this for prenorm
        #return torch.exp(x)
        return torch.softmax(x, dim = -1)


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x)
        return x

    
class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = ALiBiMultiHeadAttention(num_layers = 6, d_model = input_dim, num_heads = num_heads, max_len = input_dim, dropout = 0, expansion_factor = 1)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            #nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.prenorm = nn.LayerNorm(input_dim) 
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim) #commented out for prenorm only version
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.prenorm(x)
        attn_out = self.self_attn(x)
        x = self.norm1(x) + self.norm1(self.dropout(attn_out)) 

        linear_out = self.linear_net(x)
        x = self.norm2(x) + self.norm2(self.dropout(linear_out)) 
        return x

def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


class ALiBiMultiHeadAttention(nn.Module):
    def __init__(self, 
                 num_layers, 
                 d_model,  
                 num_heads,  
                 max_len,  
                 dropout,  
                 expansion_factor 
                ):
        
        super().__init__()
        self.num_heads = num_heads
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("m", get_alibi_slope(self.num_heads))
        self.kqv = nn.Linear(d_model, 3 * d_model, bias = False) 
        self._reset_parameters()

    def _reset_parameters(self):
        if wandb.config['model_init'] == "xavier":
            nn.init.xavier_uniform_(self.kqv.weight)
        else:
            nn.init.orthogonal_(self.kqv.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, _ = x.shape

        key, query, value = self.kqv(x).chunk(3, dim=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        bias = (self.m * get_relative_positions(seq_len).to(device)).unsqueeze(0)

        score = torch.matmul(query, key) / self.scale + bias

        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.dropout(out)
        return out


    
class Trainer:
    """Class that handles training the model"""
    def __init__(self, model_init, model_cfg, data, train_cfg, use_gpu = True):
        self.model = model_init(**model_cfg)
        self.data = data
        
        if use_gpu and torch.cuda.is_available():
            self.model.to(device)
            self.data = tuple([d.to(device) for d in self.data])
            
        self.cd_ratio = train_cfg.get('cd_ratio', 0.2)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                         lr= train_cfg.get('lr'),
                                         betas = (0.9, 0.999),
                                         weight_decay = train_cfg.get('alpha', 0.0))
        
        #self.base_optimizer = torch.optim.SGD
        #self.optimizer = SAM(self.model.parameters(), self.base_optimizer, lr = train_cfg.get('lr'), betas = (0.9, 0.999))
        
    def make_cd_mask(self, train_input, train_output):
        """Creates boolean mask for coordinated dropout.
        In coordinated dropout, a random set of inputs is zeroed out,
        and only the corresponding outputs (i.e. same trial, timestep, and neuron)
        are used to compute loss and update model weights. This prevents
        exact spike times from being directly passed through the model.
        """
        cd_ratio = self.cd_ratio
        input_mask = torch.zeros((train_input.shape[0] * train_input.shape[1] * train_input.shape[2]), dtype=torch.bool)
        idxs = torch.randperm(input_mask.shape[0])[:int(round(cd_ratio * input_mask.shape[0]))]
        input_mask[idxs] = True
        input_mask = input_mask.view((train_input.shape[0], train_input.shape[1], train_input.shape[2]))
        output_mask = torch.ones(train_output.shape, dtype=torch.bool)
        output_mask[:, :, :input_mask.shape[2]] = input_mask
        return input_mask, output_mask
    
    def train_epoch(self, epoch, scaler):
        
        num_batches = len(train_input_batches)
        train_input, train_output, val_input, val_output, *_ = self.data
        self.model.train()
        
        summed_train_loss = torch.Tensor([0])
        summed_val_loss = torch.Tensor([0])
        
        for batch in range(num_batches):
            self.model.zero_grad()           
            
            input_mask, output_mask = self.make_cd_mask(train_input[batch], train_output[batch])
            masked_train_input = train_input[batch].clone()
            masked_train_input[input_mask] = 0.0
            
            with torch.cuda.amp.autocast():
                
                train_predictions = self.model(masked_train_input)
                this_train_output = train_output[batch]
                
                loss = torch.nn.functional.poisson_nll_loss(train_predictions[output_mask], this_train_output[output_mask], log_input = False)
            
            scaler.scale(loss).backward()
            lv = loss.detach().cpu().numpy()
            
            with torch.cuda.amp.autocast():
                val_predictions = self.model(val_input)
                val_loss = torch.nn.functional.poisson_nll_loss(val_predictions, val_output, log_input = False)
                
            lv2 = val_loss.detach().cpu().numpy()
            
            summed_train_loss += lv
            summed_val_loss += lv2
            
            if batch == num_batches-1:
                avg_train_loss = summed_train_loss / num_batches
                avg_val_loss = summed_val_loss / num_batches
                
                print(f"Epoch: {epoch} train loss: {avg_train_loss[0]}          val loss: {avg_val_loss[0]}")
                
                metrics = {
                    "train/train_loss": avg_train_loss,
                    "val/val_loss": avg_val_loss
                }
                
                wandb.log({**metrics})
                
            scaler.step(self.optimizer)
            scaler.update()

    def train(self, num_epochs, save_path = None):
        """Trains model for given number of iterations"""
        train_log = []
        best_score = 1e8
        
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(num_epochs):
            self.train_epoch(epoch, scaler)
    
    def save_checkpoint(self, file_path, data):
        default_ckpt = {
            "state_dict": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }
        assert "state_dict" not in data
        assert "optim_state" not in data
        default_ckpt.update(data)
        torch.save(default_ckpt, file_path)

        


# Run parameters
dataset_name = 'mc_maze'
phase = 'train'
bin_size = 5

# Extract data
training_input, training_output, eval_input = get_data(dataset_name, phase, bin_size)


# Train/val split and convert to Torch tensors
num_train = int(round(training_input.shape[0] * 0.75))
train_input = torch.Tensor(training_input[:num_train])

train_output = torch.Tensor(training_output[:num_train])
val_input = torch.Tensor(training_input[num_train:])
val_output = torch.Tensor(training_output[num_train:])
eval_input = torch.Tensor(eval_input)

# Model hyperparameters
BATCH_SIZE = 64
DROPOUT = 0.35 
L2_WEIGHT = 0
LR_INIT = 4.5e-4
CD_RATIO = 0.25
HIDDEN_DIM = 128
USE_GPU = True

train_input_batches, train_output_batches = make_batches(train_input, train_output, batch_size = BATCH_SIZE)

RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_model'
#RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), './runs/')
#if not os.path.isdir(RUN_DIR):
#    os.mkdir(RUN_DIR)

# Train model
wandb.init(
    project = "test",
    name = "orthog init 128",
    config = {
        'input_dim': train_input.shape[2], 
        'hidden_dim': HIDDEN_DIM, 
        'output_dim': train_output.shape[2], 
        'num_heads': 2,
        'num_layers': 6,
        'lr': LR_INIT,
        'dropout': DROPOUT,
        'model_init': "orthogonal"
    }
)

runner = Trainer(
    model_init=Transformer,
    model_cfg=wandb.config,
    data=(train_input_batches, train_output_batches, val_input, val_output, eval_input),
    train_cfg={'lr': LR_INIT, 'alpha': L2_WEIGHT, 'cd_ratio': CD_RATIO},
    use_gpu=USE_GPU,
)

#model_dir = os.path.join(RUN_DIR, RUN_NAME)
#os.mkdir(os.path.join(RUN_DIR, RUN_NAME))
train_log = runner.train(num_epochs = 10, save_path = None)

wandb.finish()