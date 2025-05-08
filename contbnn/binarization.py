import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.autograd import Function
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

class BinActConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BinActConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([2.5]).float().to(device)
        self.t = torch.tensor([1]).float().to(device)

    def forward(self, input):
        w = self.weight
        a = input
        ba = BinaryQuantize().apply(a, self.k, self.t)
        output = F.conv2d(ba, w, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

class LearnableAlpha():
    """Learnable channel-wise scaling factor for convolutional layer weights."""
    def __init__(self, model, device):
        self.alphas = nn.ParameterDict()
        self.device = device

        for name, m in model.named_modules():
            if isinstance(m, BinActConv):
                clean_name = f"{name.replace('.', '_')}_weight"
                alpha_init = m.weight.data.abs().mean(dim=(1, 2, 3), keepdim=True)

                self.alphas[clean_name] = nn.Parameter(alpha_init.to(self.device), requires_grad=True)

    def get_alpha(self, name):
        clean_name = f"{name.replace('.', '_')}_weight"
        if clean_name in self.alphas:
            return self.alphas[clean_name]
        else:
            raise KeyError(f"Alpha for {name} not found in LearnableAlpha module.")
        
class Binarization():
    """Utility class to binarize a model, save and restore FP weights"""
    def __init__(self, model, device):
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        self.device = device
        for name, m in model.named_modules():
            if isinstance(m, BinActConv):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)
                self.target_modules.append((name, m.weight))
        self.bin_convs_num = len(self.target_modules)


    def binarize(self, alpha_module):
        self.save_params()
        self.binarizeConvParams(alpha_module)

    def save_params(self):
        """Saves FP weights before binarization"""
        for index in range(self.bin_convs_num):
            self.saved_params[index].copy_(self.target_modules[index][1].data)

    def binarizeConvParams(self, alpha_module):
        """Binarizes weights using channel-wise scaling factor alpha"""
        with torch.no_grad():
            for name, weight in self.target_modules:
                alpha = alpha_module.get_alpha(name).to(self.device)

                if alpha is not None:
                    weight.data.copy_(alpha * torch.sign(weight.data))
                else:
                    weight.data.copy_(torch.sign(weight.data))

    def restore(self):
        """Restores FP weights binarization"""
        for index in range(self.bin_convs_num):
            self.target_modules[index][1].data.copy_(self.saved_params[index])

def weighted_loss(original_loss, model, alpha_module, binarization_module, lambda_param):
    """Computes weighted loss with regularization term applied only to binarized model parameters."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg_term = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=False)
    bin_convs_num = binarization_module.bin_convs_num

    # Iterate over the binarized conv layers
    for name, m in model.named_modules():
        if isinstance(m, BinActConv):
            alpha = alpha_module.get_alpha(name)

            if alpha is not None:
                alpha = alpha.to(device)

                assert m.weight.requires_grad, f"Weight {name} is missing gradients!"
                assert alpha.requires_grad, f"Alpha {name} is missing gradients!"

                reg_term += torch.sum((m.weight - alpha)**2 * (m.weight + alpha)**2)

    # Scale regularization term
    reg_term = (lambda_param / bin_convs_num) * reg_term
    assert reg_term.requires_grad, f"Reg term is missing gradients!"

    # Ensure total loss tracks gradients
    total_loss = original_loss + reg_term
    assert total_loss.requires_grad, f"Total loss {total_loss} is missing gradients!"

    return total_loss, reg_term
