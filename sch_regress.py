""" Linear regression with SchNet encoders """

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

from dataset import *
from tqdm import tqdm
import matplotlib.pyplot as plt

import pickle

def get_activation(name, cache):
    def hook(model, input, output):
        if type(cache) == dict:
            if name not in cache:
                cache[name] = [output.detach()]
            else:
                cache[name].append(output.detach())
        elif type(cache) == list:
            cache.append(output.detach())
    return hook

def setup_data(data_dir: str="sch/"):
    fnames = ["sch_regress_activations.pt", "sch_regress_activations_norm.pt", "sch_regress_energy_gaps.pt"]
    if np.all([os.path.exists(os.path.join(data_dir, fname)) for fname in fnames]):
        return
    device = torch.device("cpu")
    dataset = QM9(data_dir)
    model, datasets = SchNet.from_qm9_pretrained(data_dir, dataset, 0)

    model = model.to(device)

    ds = ExcitedDataset(".", pre_transform=schnet_transform)

    activations = []
    energy_gaps = []
    model.lin1.register_forward_hook(get_activation("lin1", activations))

    for data in tqdm(ds):
        data = data.to(device)
        with torch.no_grad():
            model(data.z, data.pos, data.batch)
            energy_gaps.append(data.y.detach().cpu())

    activations = [a.sum(0) for a in activations]
    activations = torch.stack(activations, dim=0)
    energy_gaps = torch.stack(energy_gaps, dim=0).squeeze(-1)

    torch.save(activations, os.path.join(data_dir, "sch_regress_activations.pt"))
    activations /= activations.sum(0)

    torch.save(activations, os.path.join(data_dir, "sch_regress_activations_norm.pt"))
    torch.save(energy_gaps, os.path.join(data_dir, "sch_regress_energy_gaps.pt"))

class SchDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        out = self.linear(x)
        return out

def linreg(dl, Nepochs=1000):
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    model = LinearRegression(64, 1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []

    for epoch in range(Nepochs):
        if epoch % (Nepochs // 10) == 0:
            print(f"Epoch {epoch}/{Nepochs}")
        for data in dl:
            x = data[0]
            y = data[1]

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        losses.append(loss.detach().cpu())
    plt.plot(losses)
    plt.show()
    return model

if __name__ == "__main__":
    setup_data()
    activations = torch.load("sch/sch_regress_activations_norm.pt")
    gaps = torch.load("sch/sch_regress_energy_gaps.pt")

    ds = SchDataset(activations, gaps)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = linreg(dl, Nepochs=2000)

    pred = []
    actual = []
    for data in ds:
        x = data[0]
        y = data[1]
        out = model(x)
        pred.append(out.detach().cpu())
        actual.append(y.detach().cpu())
    pred = torch.stack(pred, dim=0)
    actual = torch.stack(actual, dim=0)
    torch.save(pred, "sch/pred.pt")
    torch.save(actual, "sch/actual.pt")
