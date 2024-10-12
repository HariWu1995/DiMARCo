"""
Diffusion-Model for ARCorpus
DI    -   M     -   ARCO    

Reference:
    https://huggingface.co/learn/diffusion-course/en/unit1/3
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L879
"""
import os

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


class ModelTrainer:

    def __init__( self, 
        
        train_dataloader,
        eval_dataloader = None,

        # Model
        backbone: str = 'unet',
        num_stages: int = 2, 
        num_classes: int = 10, 
        init_filters: int = 32,
    background_class: int = -1,
    upscale_bilinear: bool = True,

        # Diffusion
        objective: str = 'initial',
    noise_schedule: str = 'beta',
    diffusion_steps: int = 10, 
        eval_noise: float = 0.768,
        train_noise: float = 0.95,
        # min_noise: float = math.sqrt(eps), 
        # max_noise: float = math.sqrt(
        #                    math.sqrt(eps)),

        # Training
        loss_fn: nn.Module or str = 'mape',
        loss_reduce: str = 'mean',
        optimizer: optim.Optimizer or str = 'adam',
        optimizer_opt: Dict = dict(),
        lr_schedule: str = 'cosine',

        lr: float = 1e-4,
        num_epochs: int = 100,
        num_steps: int = 10_000,
        init_step: int = 0,
        eval_step: int = -1,
        accum_steps: int = 16,
        grad_max_norm: Union[float, List[float]] = 1.,

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

        ## Diffusion
        self.objective = objective.lower()  # prediction: target (y) / noise (n) / initial (x, default)
        self.diff_steps = diffusion_steps
        self.noischedule = alpha_schedule if noise_schedule == 'alpha' else \
                            beta_schedule
        self.train_noise = train_noise
        self.eval_noise = eval_noise

        ## Modeling
        self.build( 
                    backbone = backbone, 
                  num_stages = num_stages,
                 num_classes = num_classes,
                init_filters = init_filters,
            background_class = background_class,
            upscale_bilinear = upscale_bilinear,
        )

        ## Training
        self.num_steps = num_steps if num_epochs <= 0 else \
                         num_epochs * len(train_dataloader)
        self.train_step = init_step
        self.eval_step = eval_step
        self.accum_steps = accum_steps
        self.grad_max_norm = grad_max_norm if isinstance(grad_max_norm, (list, tuple)) \
                        else [grad_max_norm] * 2

        self.train_loader = train_dataloader
        self.eval_loader = eval_dataloader

        self.loss_fn = loss_fn if isinstance(loss_fn, nn.Module) else \
                                get_loss_fn(loss_fn=loss_fn)

        optimizer_opt.update(dict(lr=lr))
        if not isinstance(optimizer, optim.Optimizer):
            optimizer = get_optimizer(optimizer)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_opt)

        self.lr_scheduler = get_scheduler(lr_schedule, self.optimizer)
        
        ## Accelerator
        self.accelerator = Accelerator(
            split_batches = distributed_batches,
            mixed_precision = precision_type if precision_type else 'no',
        )

        self.model = self.accelerator.prepare(self.model)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.train_loader = self.accelerator.prepare(self.train_loader)
        if self.eval_loader:
            self.eval_loader = self.accelerator.prepare(self.eval_loader)

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

    def build(self, backbone, init_filters, num_stages, 
            upscale_bilinear=None, num_classes=None, background_class=None, **kwargs):

        unet_kwargs = dict( in_channels = 1, 
                            out_channels = 1, 
                            init_filters = init_filters,
                            num_stages = num_stages, )

        self.backbone = backbone.lower()

        if self.backbone == 'catunet':
            from ..models import CatUNet as UNet
            unet_kwargs.update(dict(num_classes=num_classes, 
                                background_class=background_class))

        elif self.backbone == 'dilunet':
            from ..models import DilUNet as UNet
            unet_kwargs.update(dict(upscale_bilinear=upscale_bilinear))

        elif self.backbone == 'munet':
            from ..models import mUNet as UNet

        else:
            from ..models import UNet
        
        self.model = UNet(**unet_kwargs)

    def save(self, milestone: str or int = 'latest', 
                weights_only: bool = True):

        if not self.accelerator.is_local_main_process:
            return
        
        if weights_only:
            state_dict = self.model.state_dict()
        else:
            state_dict = {
                  'version' : torch.__version__,
                     'step' : self.train_step,
                'optimizer' : self.optimizer.state_dict(),
                    'model' : self.accelerator.get_state_dict(self.model),
                   'scaler' : self.accelerator.scaler.state_dict() \
                           if self.accelerator.scaler is not None else None,
            }

        if isinstance(milestone, int):
            milestone = f"{milestone:07d}"
        torch.save(state_dict, str(self.save_folder / f'model-{milestone}.pt'))

    def load(self, milestone: str or int = 'latest', 
                weights_only: bool = True):

        device = self.accelerator.device
        model = self.accelerator.unwrap_model(self.model)

        if isinstance(milestone, int):
            milestone = f"{milestone:07d}"

        state_dict = torch.load(str(self.save_folder / f'model-{milestone}.pt'), map_location=device)

        if weights_only:
            model.load_state_dict(state_dict)
            return 

        # Load all
        model.load_state_dict(state_dict['model'])

        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.train_step = state_dict['step']

        if 'version' in state_dict:
            print(f"loading from version {state_dict['version']}")

        if (self.accelerator.scaler is not None) and (state_dict['scaler'] is not None):
            self.accelerator.scaler.load_state_dict(state_dict['scaler'])

    def preprocess(self, x, y, N):

        device = self.device

        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=1).to(device)

        # Special cases
        if self.backbone == 'catunet':
            # x can be considered as mask / magnitude of category `c`
            c = x.to(device)
            x = torch.where(x >= 0, 1., 0)            
        else:
            # normalize num_classes = 10, background = -1
            x = torch.where(x >= 0, x/10, x)

        # Add noise
        n = N * torch.rand(x.shape[0])
        x_n = self.noischedule(x, n)
        x_n = x_n.to(device)
        X = [x_n, c] if self.backbone == 'catunet' else [x_n]

        # Objective
        if self.objective == 'initial':
            y = x.to(device) if self.backbone != 'catunet' else c

        elif self.objective == 'noise':
            y = n.to(device)

        return X, y

    def train(self):
        is_main = self.accelerator.is_main_process
        device = self.device

        eval_loader  =  self.eval_loader
        train_loader = self.train_loader
        train_loader = loop(train_loader, shuffle=True)

        losses, losses_val = [inf], [inf]

        pbar = tqdm(initial = self.train_step, 
                      total = self.num_steps, disable = not is_main)

        while self.train_step < self.num_steps:

            is_1st_epoch = self.train_step < len(self.train_loader)

            ## Training phase
            self.model.train()

            total_loss = 0.
            accum_steps = self.accum_steps if not is_1st_epoch else 1
            max_gradient = self.grad_max_norm[0 if is_1st_epoch else 1]

            for _ in range(accum_steps):

                # Load batch samples
                x, y = next(train_loader)
                x, y = self.preprocess(x, y, self.train_noise)

                # Prediction
                with self.accelerator.autocast():
                    y_hat = self.model(*x, t = self.diff_steps)
                    loss  = self.loss_fn(y, y_hat)
                    loss /= self.accum_steps
                    total_loss += loss.item()

                # Backward error-propagation
                self.accelerator.backward(loss)

            losses.append(total_loss)
            pbar.set_description(f'loss: {total_loss:.4f}')

            self.accelerator.wait_for_everyone()
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_gradient)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.accelerator.wait_for_everyone()
            self.train_step += 1

            ## Evaluation phase
            loss_eval = None

            if (self.train_step % self.eval_step == 0) and (not is_1st_epoch) \
            and (self.eval_step > 0 and eval_loader is not None):

                self.model.eval()
                loss_eval = 0.

                for x, y in eval_loader:
                    x, y = self.preprocess(x, y, self.eval_noise)
                    with torch.no_grad():
                        y_hat = self.model(*x)
                        loss = self.loss_fn(y, y_hat)
                        loss_eval += loss.item()

                losses_val.append(loss_eval)

            else:
                losses_val.append(inf)

            ## LR Scheduling
            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step()
                except TypeError:
                    # for Reduce LR on Pleateau
                    self.lr_scheduler.step(total_loss if loss_eval is None \
                                                    else loss_eval)
                # print('\nUpdate lr =', self.lr_scheduler.get_last_lr())

            ## Checkpointing
            if is_main:

                # Compare loss                
                is_best_ckpt = losses[-1] < min(losses[:-1]) if not loss_eval else \
                            losses_val[-1] < min(losses_val[:-1])

                if self.save_best_ckpt and is_best_ckpt:
                    self.save(milestone='best', weights_only=False)

                if self.save_last_ckpt:
                    self.save(milestone='last', weights_only=True)

                if self.save_every_ckpt:
                    self.save(milestone=self.train_step, weights_only=True)

            pbar.update(1)

        # Save losses
        import pandas as pd
        loss_df = pd.DataFrame(dict(train_loss=losses, 
                                     eval_loss=losses_val,)).drop(index=[0]).reset_index(names=['iteration'])
        loss_df.to_csv(str(self.save_folder / 'losses.csv'), index=False)

        self.accelerator.print('Training is finished!')


