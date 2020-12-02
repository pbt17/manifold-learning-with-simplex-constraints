import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from sacred import Experiment
from scipy.sparse import dok_matrix
from sklearn.cluster import spectral_clustering
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import LocalDictionary
from utils import *

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)


ex = Experiment("Clustering Two Circles - Separation vs Dictionary Size vs Accuracy")


@ex.config
def cfg():
    verbose = True
    fixed_seed = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # for network
    num_layers = 15
    penalty = 1.0
    # for clustering
    deltas = [
        1.0,
        0.95,
        0.9,
        0.85,
        0.8,
        0.75,
        0.7,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.05,
        0.0,
    ]
    dict_sizes = [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
    ]
    test_size = 1000
    # for training
    lr = 1e-3
    batches = 2000
    batch_size = 32
    image_refresh = 2000


def sample(n, delta, even_clusters=True, noise=None):
    # sample from outer (unit) circle
    y = F.normalize(torch.randn(n, 2), dim=1)
    # scale some points to inner circle
    if even_clusters:  # each circle has equal mass
        l = torch.randint(0, 2, (n,))
    else:  # mass is proportional to length
        l = torch.bernoulli(torch.ones(n) * (1 - delta) / (2 - delta))
    y *= (1 - delta) + delta * l.view(n, 1)
    if noise:
        y += noise * torch.randn(y.shape)
    return y, l


@ex.capture
def train(
    delta,
    y_test,
    l_test,
    net,
    path,
    writer,
    verbose,
    device,
    lr,
    batches,
    batch_size,
    image_refresh,
):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = LocalDictionaryLoss(net.penalty)

    smooth_loss_reconstruction = []
    smooth_loss_penalty = []
    smooth_loss_total = []
    smooth_degree = []
    short_circuit_now = False
    short_circuit_lag = 1000

    for batch in tqdm(range(batches), disable=not verbose):
        # draw fresh sample from infinite training data set
        y, l = sample(batch_size, delta)
        y = y.to(device)
        # forward pass
        x_hat = net.encode(y)
        # backward pass
        optimizer.zero_grad()
        loss_reconstruction, loss_penalty, loss_total = criterion.forward_detailed(
            net.W, y, x_hat
        )
        loss_total.backward()
        optimizer.step()
        # logging
        writer.add_scalar(
            "loss (reconstruction)", loss_reconstruction.item(), global_step=batch
        )
        writer.add_scalar("loss (penalty)", loss_penalty.item(), global_step=batch)
        writer.add_scalar("loss (total)", loss_total.item(), global_step=batch)
        degree = x_hat.bool().float().sum(dim=1).mean()
        writer.add_scalar("degree", degree.item(), global_step=batch)
        # smoothed averages
        smooth_loss_reconstruction.append(loss_reconstruction.item())
        smooth_loss_penalty.append(loss_penalty.item())
        smooth_loss_total.append(loss_total.item())
        smooth_degree.append(degree.item())
        if batch > 0:
            for metric in (
                smooth_loss_reconstruction,
                smooth_loss_penalty,
                smooth_loss_total,
                smooth_degree,
            ):
                metric[-1] = 0.99 * metric[-2] + 0.01 * metric[-1]
        # stop early?
        if batch > short_circuit_lag:
            short_circuit_now = True
            for metric in (
                smooth_loss_reconstruction,
                smooth_loss_penalty,
                smooth_loss_total,
                smooth_degree,
            ):
                ratio = metric[-1] / metric[-short_circuit_lag]
                short_circuit_now &= 0.99 < ratio < 1.01
        # visualize
        if short_circuit_now or (batch + 1) % image_refresh == 0:
            with torch.no_grad():
                x_hat_test = net.encode(y_test)
                y_hat_test = net.decode(x_hat_test)
            plt.figure(figsize=(5, 5))
            plt.scatter(*y_test.cpu().T, c="gray", s=0.5)
            plt.scatter(*y_hat_test.cpu().clone().detach().T, c=l_test, s=5.0)
            plt.scatter(*net.W.data.cpu().T, c="red")
            plt.xticks([], [])
            plt.yticks([], [])
            plt.axis("equal")
            plt.tight_layout()
            cur_path = path / "figs"
            cur_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(cur_path / f"sep{delta}_dict{net.hidden_size.item()}.png")
            plt.close()
        # stop early?
        if short_circuit_now:
            break


@ex.automain
def run(
    _run,
    verbose,
    fixed_seed,
    device,
    num_layers,
    penalty,
    deltas,
    dict_sizes,
    test_size,
):
    # for figures, tensorboard, etc.
    if not _run.observers:
        raise Exception("no output directory specified (use the -F flag)")
    else:
        path = Path(_run.observers[0].dir)

    # view config in tensorboard
    writer = SummaryWriter(path)
    writer.add_text("config", str(_run.config))
    writer.close()

    # for reproducibility
    if fixed_seed:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
    n = test_size

    # run the experiment
    test_accuracy = pd.DataFrame()
    for delta in deltas:
        # fix test data in advance
        y_test, l_test = sample(n, delta)
        y_test = y_test.to(device)
        for m in dict_sizes:
            if verbose:
                print(f"Separation = {delta}, Dictionary Size = {m}")
            # initialize network
            net = LocalDictionary(num_layers, 2, m, penalty)
            net.W.data = sample(m, delta)[0]
            net.step.data.fill_(net.W.data.svd().S[0] ** -2)
            net = net.to(device)
            # train the network
            writer = SummaryWriter(
                path / "logs" / f"sep{delta}_dict{net.hidden_size.item()}"
            )
            train(delta, y_test, l_test, net, path, writer)
            writer.close()
            # pass in test data
            with torch.no_grad():
                x_hat_test = net.encode(y_test)
                y_hat_test = net.decode(x_hat_test)
            # perform clustering
            sim = dok_matrix((n + m, n + m), dtype=np.float32)
            for i, j in tqdm(x_hat_test.nonzero()):
                sim[i, n + j] = x_hat_test[i, j].item() / 2
                sim[n + j, i] = x_hat_test[i, j].item() / 2
            l_hat_test = spectral_clustering(sim, n_clusters=2)
            acc_test = acc(l_test, l_hat_test[:n], 2)[0]
            test_accuracy = test_accuracy.append(
                {"Separation": delta, "Dictionary Size": m, "Value": acc_test},
                ignore_index=True,
            )
            print(f"accuracy[{delta}, {m}] = {acc_test}")

    # record accuracy
    json.dump(
        test_accuracy.to_numpy().tolist(), open(path / "accuracy.json", "w"), indent=4
    )

    # plot accuracy heatmap
    test_accuracy = test_accuracy.pivot(
        index="Dictionary Size", columns="Separation", values="Value"
    )
    test_accuracy.index = test_accuracy.index.astype(np.int32)
    plt.clf()
    sns.heatmap(test_accuracy, annot=True, cmap="viridis")
    plt.tight_layout()
    plt.savefig(path / "heatmap.png")
