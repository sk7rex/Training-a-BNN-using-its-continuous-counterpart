import torch
import torch.nn as nn
import pickle
import os

from contbnn import get_cifar, train_binary_nn, plot_lambda, training_plot, plot_weights_distribution, plot_alpha_distribution
from vggsmall_model import VGG_SMALL

def lambda_schedule_soft(epoch, num_epochs):
    return 30 ** (epoch / num_epochs)

device = "cuda" if torch.cuda.is_available() else "cpu"
vgg = VGG_SMALL()
train_dataloader, val_dataloader = get_cifar()
loss_fn = nn.CrossEntropyLoss().to(device)
num_epochs = 2
initial_lr = 0.1


model, losses, accs, alpha_module = train_binary_nn(vgg, loss_fn, train_dataloader, val_dataloader, num_epochs, initial_lr, lambda_schedule_soft, tau = 5, eps = 0.1)

data = [(model, "model.pkl"), (losses, "losses.pkl"), (accs, "accs.pkl"), (alpha_module, "alpha_module.pkl")]
save_path = '/cifar/cifar_vgg'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

for elem, filename in data:
    full_path = os.path.join(save_path, filename)
    with open(full_path, 'wb') as file:
        pickle.dump(elem, file)

plot_lambda(lambda_schedule_soft, num_epochs, save_path)
training_plot(losses, accs, save_path)
plot_weights_distribution(model, save_path, threshold=0.2)
plot_alpha_distribution(alpha_module, save_path)