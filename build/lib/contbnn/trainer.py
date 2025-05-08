import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Function
from tqdm import tqdm
import math
import numpy as np

from .utils import acc
from .binarization import LearnableAlpha, Binarization, weighted_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
use_gpu = torch.cuda.is_available()

def train_binary_nn(model, loss_fn, train_dataloader, val_dataloader, num_epochs, initial_lr, lambda_schedule, eps=0.5, tau=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    alpha_module = LearnableAlpha(model, device)
    optimizer = optim.SGD(list(model.parameters()) + list(alpha_module.alphas.values()), lr=initial_lr)
    binarization_module = Binarization(model, device)

    losses = {'train': [], "val": [], 'base': [], 'reg': []}
    accs = {'train': [], 'val': []}

    for epoch in tqdm(range(num_epochs)):
        lambda_param = lambda_schedule(epoch, num_epochs)

        train_loss, train_acc, train_reg_term, train_base_loss = train(model, loss_fn, train_dataloader, optimizer, binarization_module, alpha_module, lambda_param, initial_lr, tau, eps)
        if train_loss == None:
            return model, losses, accs, alpha_module

        losses['train'].append(train_loss)
        losses['base'].append(train_base_loss)
        losses['reg'].append(train_reg_term)
        accs['train'].append(train_acc)

        val_loss, val_acc = validate(model, alpha_module, val_dataloader, binarization_module, loss_fn)

        losses['val'].append(val_loss)
        accs['val'].append(val_acc)

        print("\n", f"Epoch {epoch + 1}: Loss = {losses['train'][-1]}, Accuracy = {accs['val'][-1]}")
        print(f"Epoch {epoch + 1}: Base loss = {losses['base'][-1]}, Reg term = {losses['reg'][-1]}")

    return model, losses, accs, alpha_module

def train(model, loss_fn, train_dataloader, optimizer, binarization_module, alpha_module, lambda_param, initial_lr, tau, eps):
    model.train()

    train_losses = 0
    train_accs = 0
    iter_count = 0
    train_reg_term = 0
    train_base_loss = 0

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        original_loss = loss_fn(outputs, targets)
        total_loss, reg_term = weighted_loss(original_loss, model, alpha_module, binarization_module, lambda_param)
        train_reg_term += reg_term.item()
        train_base_loss += original_loss.item()
        if torch.isnan(original_loss) or torch.isnan(reg_term):
            return None, None, None, None

        total_loss.backward()

        alpha_k = initial_lr
        binarization_module.binarize(alpha_module)
        with torch.no_grad():
            bin_outputs_old = model(inputs)
            loss_old = loss_fn(bin_outputs_old, targets)
        binarization_module.restore()
        flag = False
        for k in range(tau):
            with torch.no_grad():
                for param in model.parameters():
                    param -= alpha_k * param.grad  # In-place update
                for alpha in alpha_module.alphas.values():
                    alpha -= alpha_k * alpha.grad

                binarization_module.binarize(alpha_module)
                with torch.no_grad():
                    bin_outputs_new = model(inputs)
                    loss_new = loss_fn(bin_outputs_new, targets)
                binarization_module.restore()

                if loss_new < loss_old:
                    flag = True
                    break  # Accept the update
                else:
                    for param in model.parameters():
                        param += alpha_k * param.grad  # Restore previous weights if rejected
                    for alpha in alpha_module.alphas.values():
                        alpha += alpha_k * alpha.grad
                    # Reduce learning rate if condition not met
                    alpha_k *= eps

        if flag:
            train_losses += loss_new.item()
        else:
            train_losses += loss_old.item()

        train_accs += acc(torch.argmax(bin_outputs_new, dim=1), targets)
        iter_count += 1

    return train_losses / iter_count, train_accs / iter_count, train_reg_term / iter_count, train_base_loss / iter_count

def validate(model, alpha_module, val_dataloader, binarization_module, loss_fn):
    model.eval()
    binarization_module.binarize(alpha_module)

    val_accs = 0
    val_losses = 0
    iter_count = 0

    for inputs, targets in val_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            val_losses += loss.item()
            val_accs += acc(torch.argmax(outputs, dim=1), targets)
            iter_count += 1

    binarization_module.restore()

    return val_losses / iter_count, val_accs / iter_count
