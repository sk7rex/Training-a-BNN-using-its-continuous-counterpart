import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_lambda(lambda_schedule, num_epochs, save_path):
    plt.figure(figsize=(10, 6))
    x = np.array(list(range(1, num_epochs + 1)))
    y = lambda_schedule(x, num_epochs)
    plt.plot(x, y, label="Lambda(epoch)")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Lambda value')
    plt.title('Lambda Scheduling')
    
    filename = 'lambda.png'
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, bbox_inches='tight')
    print(f"Figure saved to {full_path}")

def training_plot(losses, accs, save_path):
    fig, axs = plt.subplots(3, 2)

    axs[0, 0].plot(losses['train'], label = "Train Loss")
    axs[0, 0].legend()
    axs[0, 1].plot(accs['train'], label = "Train Accuracy", c='orange')
    axs[0, 1].legend()
    axs[1, 0].plot(losses['val'], label = "Validation Loss")
    axs[1, 0].legend()
    axs[1, 1].plot(accs['val'], label = "Validation Accuracy", c='orange')
    axs[1, 1].legend()
    axs[2, 0].plot(losses['base'], label = "Base Loss")
    axs[2, 0].legend()
    axs[2, 1].plot(losses['reg'], label = "Regularization Term")
    axs[2, 1].legend()
    filename = 'training.png'
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, bbox_inches='tight')
    print(f"Figure saved to {full_path}")

# Function to plot weights distribution (excluding first and last layers)
def plot_weights_distribution(model, save_path, threshold=0.2):
    weights = []
    for name, param in model.named_parameters():
        if isinstance(param, torch.nn.parameter.Parameter):
                if torch.any(torch.isnan(param)):
                    print(f"NaN detected in {name}, skipping")
                    continue
                weights.append(param.detach().cpu().numpy().flatten())

    if weights:
        weights = np.concatenate(weights)
        weights = weights[np.abs(weights) < threshold]
        plt.figure(figsize=(10, 6))
        plt.hist(weights, bins=1000, color='green', alpha=0.7)
        plt.title('Weights Distribution')
        plt.xlabel('Weight Values')
        plt.ylabel('Frequency')
        plt.grid(True)
        filename = 'weights.png'
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"Figure saved to {full_path}")
    else:
        print("No valid weights to plot.")

def plot_alpha_distribution(alpha_module, save_path):
    """Plots the distribution of alpha values."""
    alphas = np.concatenate([alpha.view(-1).cpu().detach().numpy() for alpha in alpha_module.alphas.values()])
    plt.figure(figsize=(10, 6))
    plt.hist(alphas, bins=1000, edgecolor='green')
    plt.xlabel('Alpha values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Alpha Values Across Layers')
    filename = 'scaling.png'
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, bbox_inches='tight')
    print(f"Figure saved to {full_path}")

def get_cifar():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)

    return trainloader, valloader

def get_mnist(batch_size = 128):
    normalize = torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))

    mnist_train = torchvision.datasets.MNIST(
        '/data/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    )
    mnist_test = torchvision.datasets.MNIST(
        '/data/',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    )

    train_loader = DataLoader(mnist_train, batch_size=batch_size)
    val_loader = DataLoader(mnist_test, batch_size=batch_size)

    return train_loader, val_loader

def acc(output, target):
    with torch.no_grad():
        correct = (output == target).sum().item()
        total = target.size(0)
        acc = correct / total
        return acc