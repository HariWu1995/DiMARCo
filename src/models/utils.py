from functools import partial
from typing import Union, List, Tuple, Optional

import torch
from torch import nn

from .layers.lora import LinearLoRA
from .layers.dora import LinearDoRA


def freeze_layers(model, freeze_types: Union[nn.Module, 
                                       Tuple[nn.Module]] = nn.Linear):
    for child in model.children():
        if isinstance(child, freeze_types):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze layers in children modules
            freeze_layers(child, freeze_types)


def apply_LoRA_to_Linear(model, rank: int = 2, 
                              weight: float = 1., 
                            use_DoRA: bool = False):

    layer_lora = LinearLoRA if not use_DoRA else LinearDoRA
    assign_lora = partial(layer_lora, rank=rank, weight=weight)
    
    for name, module in model.named_children():

        # If the module is a nn.Linear, replace it with LinearLoRA
        if isinstance(module, nn.Linear):
            linear_lora = assign_lora(module)
            setattr(model, name, linear_lora)

        # Recursively apply to children modules
        elif len(list(module.children())) > 0:
            apply_LoRA_to_Linear(module, rank=rank, weight=weight, use_DoRA=use_DoRA)
    
    return model


if __name__ == "__main__":

    model = nn.Sequential(
            nn.Linear(10, 100), nn.ReLU(),
            nn.Linear(100, 200), nn.ReLU(),
            nn.Linear(200, 15), nn.ReLU(),
        )
    # print(model._modules)

    # Apply LoRA
    # model._modules['0'] = LinearLoRA(model._modules['0'], rank=4, lora_weight=1)
    # model._modules['2'] = LinearLoRA(model._modules['2'], rank=4, lora_weight=2)
    # model._modules['4'] = LinearLoRA(model._modules['4'], rank=4, lora_weight=3)
    model = apply_LoRA_to_Linear(model, use_DoRA=True)
    freeze_layers(model, nn.Linear)

    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

