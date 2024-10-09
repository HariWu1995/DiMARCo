"""
Diffusion-Model for ARCorpus with Optical Flow as Guidance
DI    -   M     -   ARCO     -    PO    -  LO
"""
import math

import torch
from torch import nn

from .dimarco import DiMARCo
from .layers import GridConv
from ..const import eps


class DiMARCoPolo(DiMARCo):

    pass


