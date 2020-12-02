import random
from math import sqrt

import torch
import torch.nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as lsa
from torchvision import datasets, transforms


class LocalDictionaryLoss(torch.nn.Module):
    def __init__(self, penalty):
        super(LocalDictionaryLoss, self).__init__()
        self.penalty = penalty

    def forward(self, A, y, x):
        return self.forward_detailed(A, y, x)[2]

    def forward_detailed(self, A, y, x):
        weight = (y.unsqueeze(1) - A.unsqueeze(0)).pow(2).sum(dim=2)
        a = 0.5 * (y - x @ A).pow(2).sum(dim=1).mean()
        b = (weight * x).sum(dim=1).mean()
        return a, b, a + b * self.penalty

    
def generate_dictionary(hidden_size, input_size):
    W = torch.randn(hidden_size, input_size)
    W = F.normalize(W, dim=1)
    return W


def generate_data(num_samples, W, how="normal"):
    num_groups, group_size, input_size = W.shape
    y, x, k = [], [], []
    for i in range(num_samples):
        group_idx = random.randint(0, num_groups - 1)
        if how == "normal":
            code = torch.randn(group_size)
        elif how[0] == "uniform":
            code = torch.rand(group_size) * (how[2] - how[1]) + how[1]
        k.append(group_idx)
        x.append(code)
        y.append(code @ W[group_idx])
    y = torch.stack(y)
    x = torch.stack(x)
    k = torch.tensor(k)
    return y, x, k


def generate_normal_like(num_samples, data):
    m, n = data.shape
    mean = data.mean(dim=0)
    data = (data - mean) / sqrt(m)
    u, s, v = data.svd()
    return 1.0 * torch.randn(num_samples, n) @ (s.diag() @ v.T) + mean


def get_loader(dataset, batch_size=1, shuffle=True, num_workers=0):
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=shuffle, num_workers=num_workers
    )


def get_dataset(*tensors):
    return torch.utils.data.TensorDataset(*tensors)


def mnist_loaders(root, num_workers=4, batch_size=256):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1).float() / 255.0)]
    )
    train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return train, test


def acc(targets, predictions, k=None):
    assert len(targets) == len(predictions)
    # targets = torch.tensor(targets)
    # predictions = torch.tensor(predictions)
    n = len(targets)
    if k is None:
        k = targets.max() + 1
    cost = torch.zeros(k, k)
    for i in range(n):
        cost[targets[i].item(), predictions[i].item()] -= 1
    stuff = -cost[lsa(cost)]
    total = stuff.sum().item() / n
    for i in range(k):
        stuff[i] /= -cost[i].sum()
    return total, stuff
