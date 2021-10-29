# coding=utf-8
"""PyTorch optimization."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR



class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_step_size = warmup_steps
        self.t_total_size = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step_size):
        if step_size < self.warmup_step_size:
            return float(step_size) / float(max(1, self.warmup_step_size))
        return max(float(self.t_total_size - step_size) / float(max(1.0, self.t_total_size - self.warmup_step_size)), 0.0)


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1]  < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta_1, beta_2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta_1).add_(1.0 - beta_1, grad)
                exp_avg_sq.mul_(beta_2).addcmul_(1.0 - beta_2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_len = group['lr']
                if group['correct_bias']:  
                    bias_correction_1 = 1.0 - beta_1 ** state['step']
                    bias_correction_2 = 1.0 - beta_2 ** state['step']
                    step_len = step_len * math.sqrt(bias_correction_2) / bias_correction_1

                p.data.addcdiv_(-step_len, exp_avg, denom)
                
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)
        return loss


