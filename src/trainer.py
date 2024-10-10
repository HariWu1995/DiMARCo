from typing import Dict, Union
from pathlib import Path

import math

import torch
import torch.nn as nn
import torch.optim as optim

from accelerator import Accelerator

from .optimizers import get_optimizer
from .losses import get_loss_fn
from .const import eps


def cycle(dataset, shuffle: bool = True):
    while True:
        dataset.shuffle()
        for data in dataset:
            yield data


class ModelTrainer:

    def __init__( self, 
        
        model: nn.Module,
        dataset,

        # Training
        loss_fn: nn.Module or str,
        optimizer: optim.Optimizer or str,
        optimizer_opt: Dict = dict(),

        lr: float = 1e-4,
        num_epochs: int = -1,
        num_steps: int = 10_000,
        grad_max_norm: float = 1.,

        save_folder: str or Path = Path('./results'),
        save_best_ckpt: bool = True,
        save_last_ckpt: bool = True,
        save_every_ckpt: bool = False,

        # Distributed
        precision_type: str = 'fp16',
        grad_accum_steps: int = 1,
        distributed_batches: bool = True,

        # Others
        **kwargs
    ):

        self.grad_accum_steps = grad_accum_steps
        self.grad_max_norm = grad_max_norm
        self.num_steps = num_epochs * len(dataset) if num_epochs > 0 else num_steps
        self.step = 0
        
        ## Training
        self.model = model
        self.dataset = dataset

        self.loss_fn = loss_fn if isinstance(loss_fn, nn.Module) else \
                                get_loss_fn(loss_fn=loss_fn)

        self.optimizer = optimizer if isinstance(optimizer, optim.Optimizer) else \
                                    get_optimizer(optimizer)(model.parameters(), lr=lr, **optimizer_opt)
        
        ## Accelerator
        self.accelerator = Accelerator(
            split_batches = distributed_batches,
            mixed_precision = precision_type if precision_type else 'no',
        )

        self.model = self.accelerator.prepare(self.model)
        self.dataset = self.accelerator.prepare(self.dataset)
        self.optimizer = self.accelerator.prepare(self.optimizer)

        ## Resulting
        self.save_folder = Path(save_folder) if not isinstance(save_folder, Path) else Path
        self.save_best_ckpt = save_best_ckpt
        self.save_last_ckpt = save_last_ckpt
        self.save_every_ckpt = save_every_ckpt

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone: str or int = 'latest', 
                weights_only: bool = True):

        if not self.accelerator.is_local_main_process:
            return
        
        if weights_only:
            state_dict = self.model.state_dict()
        else:
            state_dict = {
                  'version' : torch.__version__,
                     'step' : self.step,
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
        self.step = state_dict['step']

        if 'version' in state_dict:
            print(f"loading from version {state_dict['version']}")

        if (self.accelerator.scaler is not None) and (state_dict['scaler'] is not None):
            self.accelerator.scaler.load_state_dict(state_dict['scaler'])

    def train(self):

        device = self.device
        dataset = cycle(self.dataset, shuffle=True)
        is_main_proc = self.accelerator.is_main_process

        with tqdm(initial = self.step, 
                    total = self.num_steps, disable = not is_main_proc) as pbar:

            self.model.train()

            total_loss = 0.

            for _ in range(self.grad_accum_steps):

                x, y = next(dataset)
                x, y = x.to(device), y.to(device)

                with self.accelerator.autocast():
                    y_hat = self.model(x)
                    loss = self.loss_fn(y, y_hat)
                    loss = loss / self.grad_accum_steps
                    total_loss += loss.item()

                self.accelerator.backward(loss)

            pbar.set_description(f'loss: {total_loss:.4f}')

            self.accelerator.wait_for_everyone()
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.accelerator.wait_for_everyone()
            self.step += 1

            if is_main_proc:
                self.save(milestone='last', weights_only=True)

                if self.save_every_ckpt:
                    self.save(milestone=self.step, weights_only=True)

                # TODO: compare loss
                is_best_ckpt = False
                if self.save_best_ckpt and is_best_ckpt:
                    self.save(milestone='best', weights_only=False)

            pbar.update(1)

        accelerator.print('training complete')


