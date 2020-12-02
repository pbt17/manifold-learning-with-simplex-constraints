import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalDictionary(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, penalty):
        super(LocalDictionary, self).__init__()
        self.register_buffer("num_layers", torch.tensor(num_layers))
        self.register_buffer("input_size", torch.tensor(input_size))
        self.register_buffer("hidden_size", torch.tensor(hidden_size))
        self.register_buffer("penalty", torch.tensor(penalty))
        W = torch.zeros(self.hidden_size, self.input_size)
        self.register_parameter("W", torch.nn.Parameter(W))
        step = torch.tensor(0.0)
        self.register_parameter("step", torch.nn.Parameter(step))

    def forward(self, y):
        x = self.encode(y)
        y = self.decode(x)
        return y

    def encode(self, y):
        return self.encode_accelerated(y)

    def encode_basic(self, y):
        x = torch.zeros(y.shape[0], self.hidden_size, device=y.device)
        weight = (y.unsqueeze(1) - self.W.unsqueeze(0)).pow(2).sum(dim=2)
        for layer in range(self.num_layers):
            grad = (x @ self.W - y) @ self.W.T
            grad = grad + weight * self.penalty
            x = self.activate(x - grad * self.step)
        return x

    def encode_accelerated(self, y):
        x_tmp = torch.zeros(y.shape[0], self.hidden_size, device=y.device)
        x_old = torch.zeros(y.shape[0], self.hidden_size, device=y.device)
        weight = (y.unsqueeze(1) - self.W.unsqueeze(0)).pow(2).sum(dim=2)
        for layer in range(self.num_layers):
            grad = (x_tmp @ self.W - y) @ self.W.T
            grad = grad + weight * self.penalty
            x_new = self.activate(x_tmp - grad * self.step)
            x_tmp = x_new + layer / (layer + 3) * (x_new - x_old)
            x_old = x_new
        return x_new

    def decode(self, x):
        return x @ self.W

    def activate(self, x):
        m, n = x.shape
        cnt_m = torch.arange(m, device=x.device)
        cnt_n = torch.arange(n, device=x.device)
        u = x.sort(dim=1, descending=True).values
        v = (u.cumsum(dim=1) - 1) / (cnt_n + 1)
        w = v[cnt_m, (u > v).sum(dim=1) - 1]
        return (x - w.view(m, 1)).relu()
