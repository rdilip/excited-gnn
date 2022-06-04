import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.utils import from_smiles

from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

import msgpack


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(9, 16)
        self.conv2 = GCNConv(16, 10)
        self.fc1 = nn.Linear(10, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc1(x)
        
        return x


if __name__ == '__main__':
    unpacker = msgpack.Unpacker(open("0435.msgpack", "rb"), strict_map_key=False)
    data = next(unpacker)

    dataset = []
    excited_energies = []
    for k in data.keys():
        dataset.append(from_smiles(data[k]['species']['smiles']))
        excited_energies.append(data[k]['props']['excitedstates'][0]['energy'])
        dataset[-1].y = torch.Tensor([[excited_energies[-1]]])
        dataset[-1].x = dataset[-1].x.float()

    dl = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("mps")

    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5.e-4)
    model.train()

    for epoch in range(10):
        print(epoch)
        for batch in dl:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
