"""
Diffusion-Model for ARCorpus
DI    -   M     -   ARCO    

Reference:
    https://huggingface.co/learn/diffusion-course/en/unit1/3
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L879
"""
import os
import pandas as pd

from typing import Dict, List, Union
from pathlib import Path
from tqdm.auto import tqdm

import math

import torch
import torch.nn as nn
import torch.optim as optim

from accelerate import Accelerator

from ..optimizers import get_optimizer
from ..losses import get_loss_fn
from ..noise import alpha_schedule, beta_schedule
from ..const import eps, inf

from .utils import loop
from .callbacks import EarlyStopping
from .lr_schedulers import get_scheduler


class LoraTrainer:

    def __init__( self, 

        task_id: str,
        task_loader,
        
        # Base-Model
        model,
        backbone: str,
        adapter: str = 'lora',
        adapter_rank: int = 2,
        adapter_weight: float = 1.,
        layered_input: bool = False,

        # Diffusion
        objective: str = 'initial',
    noise_schedule: str = 'beta',
    denoising_steps: int = 10, 
        eval_noise: float = 0.768,
        train_noise: float = 0.95,
        # min_noise: float = math.sqrt(eps), 
        # max_noise: float = math.sqrt(
        #                    math.sqrt(eps)),

        # Training
        loss_fn: str = 'mape',
        loss_reduce: str = 'mean',
        optimizer: str = 'adam',
        optimizer_opt: Dict = dict(),
        lr_schedule: str = 'cosine',
        lr: float = 1e-4,
        grad_max_norm: float = 1.,

        num_steps: int = 10,
        init_step: int = 0,
        eval_step: int = 1,

        save_folder: str or Path = Path('./results'),
        save_best_ckpt: bool = True,
        save_last_ckpt: bool = False,
        save_every_ckpt: bool = False,

        # Distributed
        precision_type: str = 'fp16',
        distributed_batches: bool = True,

        # Others
        **kwargs
    ):

        self.task_id     = task_id
        self.task_loader = task_loader

        ## Diffusion
        self.denoissteps = denoising_steps
        self.noischedule = alpha_schedule if noise_schedule == 'alpha' else \
                            beta_schedule
        self.train_noise = train_noise
        self.eval_noise = eval_noise

        ## Modeling
        self.backbone = backbone
        self.layered_input = layered_input

        self.prepare(model, adapter, rank = adapter_rank, 
                                    weight = adapter_weight)

        ## Training
        self.num_steps = num_steps
        self.eval_step = eval_step
        self.train_step = init_step

        self.loss_mask = loss_fn in ['focal-layered','mse-layered','boundary-mse']
        self.loss_fn = loss_fn if isinstance(loss_fn, nn.Module) else \
                                get_loss_fn(loss_fn=loss_fn)

        optimizer_opt.update(dict(lr=lr))
        if not isinstance(optimizer, optim.Optimizer):
            optimizer = get_optimizer(optimizer)
        self.optimizer = optimizer(self.params.values(), **optimizer_opt)

        self.lr_scheduler = get_scheduler(lr_schedule, self.optimizer)
        self.grad_max_norm = grad_max_norm
        
        ## Accelerator
        self.accelerator = Accelerator(
            split_batches = distributed_batches,
            mixed_precision = precision_type if precision_type else 'no',
        )

        self.model = self.accelerator.prepare(self.model)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.task_loader = self.accelerator.prepare(self.task_loader)

        ## Checkpoint
        self.save_best_ckpt = save_best_ckpt
        self.save_last_ckpt = save_last_ckpt
        self.save_every_ckpt = save_every_ckpt
        self.save_folder = Path(save_folder) if not isinstance(save_folder, Path) \
                            else save_folder
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

    @property
    def device(self):
        return self.accelerator.device

    def prepare(self, model, adapter: str = 'lora', 
                                rank: int = 2, 
                            weight: float = 1., **kwargs):

        from src.models.utils import freeze_layers, apply_LoRA_to_Linear

        lora_config = dict(
               rank = rank, 
             weight = weight, 
            use_DoRA = adapter.lower() == 'dora'
        )

        self.model = apply_LoRA_to_Linear(model, **lora_config)
        freeze_layers(self.model, freeze_types = (nn.Conv2d, nn.Linear, nn.GroupNorm))

        self.params = {name: param for name, param in self.model.named_parameters() 
                                    if param.requires_grad}

    def trainable_params(self):
        print('-'*19)
        for name, param in self.model.named_parameters():
            print(f"{name:<69}: {param.requires_grad}")
        print('-'*19)
        for name, param in self.params.items():
            print(f"{name:<69}: {param.requires_grad}")
        quit()

    def save(self, milestone: str or int = 'latest'):

        if not self.accelerator.is_local_main_process:
            return
        
        if isinstance(milestone, int):
            milestone = f"{milestone:07d}"
        lora_path = str(self.save_folder / f'{self.task_id}-{milestone}.pt')
        lora_weights = self.params

        torch.save(lora_weights, lora_path)

    def load(self, milestone: str or int = 'latest'):

        device = self.accelerator.device
        model = self.accelerator.unwrap_model(self.model)

        if isinstance(milestone, int):
            milestone = f"{milestone:07d}"

        lora_path = str(self.save_folder / f'{self.task_id}-{milestone}.pt')
        lora_weights = torch.load(lora_path, map_location=device)

        model.load_state_dict(lora_weights, strict=False)

    def preprocess(self, x, y, N):

        device = self.device

        # 3D grid
        if self.layered_input:
            x, m = x
            x = x.to(torch.float)
            m = m.to(torch.float).to(device)
            y = y.to(torch.float).to(device)
        
        # 2D grid
        else:
            m = None
            x = x.unsqueeze(dim=1)
            y = y.unsqueeze(dim=1).to(device)

            if self.backbone == 'catunet':
                # x can be considered as mask / magnitude of category `c`
                c = x.to(device)
                x = torch.where(x >= 0, 1., 0)
            else:
                # normalize num_classes = 10, background = -1
                x = torch.where(x >= 0, x/10, x)

        # Add noise
        n = N * torch.rand(x.shape[0])
        x_n = self.noischedule(x, n).to(device)

        if self.layered_input:
            X = [x_n]
        else:
            X = [x_n, c] if self.backbone == 'catunet' else [x_n]

        return X, y, m

    def train(self):
        is_main = self.accelerator.is_main_process
        device = self.accelerator.device
        dloader = self.task_loader
        max_gradient = self.grad_max_norm

        losses, losses_val = [inf], [inf]

        pbar = tqdm(initial = self.train_step, 
                      total = self.num_steps, disable = not is_main)

        while self.train_step < self.num_steps:

            ## Training phase
            self.model.train()

            # Load batch samples
            x, y = dloader['train']
            x, y, m = self.preprocess(x, y, self.train_noise)

            # Prediction
            with self.accelerator.autocast():
                y_hat = self.model(*x, t=self.denoissteps)
                loss  = self.loss_fn(y_hat, y, m) if self.loss_mask else \
                        self.loss_fn(y_hat, y)

            # Backward error-propagation
            self.accelerator.backward(loss)

            loss_train = loss.item()
            losses.append(loss_train)
            pbar.set_description(f'loss: {loss_train:.5f} - evaloss: {min(losses_val):.5f}')

            self.accelerator.wait_for_everyone()
            self.accelerator.clip_grad_norm_(self.params.values(), max_gradient)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.accelerator.wait_for_everyone()
            self.train_step += 1

            ## Evaluation phase
            if (self.train_step % self.eval_step == 0) \
            and (self.eval_step > 0):

                self.model.eval()

                x, y = dloader['test']
                X, y, m = self.preprocess(x, y, self.eval_noise)

                with torch.no_grad():
                    y_hat = self.model(*X, t = self.denoissteps)
                    loss = self.loss_fn(y, y_hat, m) if self.loss_mask else \
                            self.loss_fn(y, y_hat)
                    loss_eval = loss.item()

                losses_val.append(loss_eval)

            else:
                losses_val.append(inf)
                loss_eval = None

            ## LR Scheduling
            loss_step = loss_eval if not loss_eval else loss_train
            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step()
                except TypeError:
                    # for Reduce LR on Pleateau
                    self.lr_scheduler.step(loss_step)
                # print('\nUpdate lr =', self.lr_scheduler.get_last_lr())

            ## Checkpointing
            if is_main:

                # Compare loss                
                is_best_ckpt = losses[-1] < min(losses[:-1]) if not loss_eval else \
                            losses_val[-1] < min(losses_val[:-1])

                if self.save_best_ckpt and is_best_ckpt:
                    self.save(milestone='best')

                if self.save_last_ckpt:
                    self.save(milestone='last')

                if self.save_every_ckpt:
                    self.save(milestone=self.train_step)

            pbar.update(1)

        # Save losses
        loss_df = pd.DataFrame(dict(train_loss=losses, 
                                     eval_loss=losses_val,)).drop(index=[0]).reset_index(names=['iteration'])
        loss_df.to_csv(str(self.save_folder / f'{self.task_id}-losses.csv'), index=False)

    def print(self, message: str, level: str = 'INFO'):
        self.accelerator.print(f"[{level}] {message}")

    def get_best_result(self):
        loss_df = pd.read_csv(str(self.save_folder / f'{self.task_id}-losses.csv'))
        loss_train = loss_df['train_loss'].min()
        loss_eval  = loss_df[ 'eval_loss'].min()
        return loss_train, loss_eval
