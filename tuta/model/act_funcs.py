# -*- coding: utf-8 -*-
""" Activation Functions """

import math
import torch


# %% activation functions
def gelu(x):
    return (torch.erf(x/math.sqrt(2.0)) + 1.0) * x * 0.5


ACT_FCN = {"gelu": gelu}
