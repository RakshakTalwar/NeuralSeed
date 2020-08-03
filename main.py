from typing import Optional, Union, Sequence, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coffee_n_sugar import CoffeeNSugar


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cns_train, cns_val = CoffeeNSugar(split='train', train_fraction=.7), CoffeeNSugar(split='val', train_fraction=.7)
dataloader_train = DataLoader(cns_train, batch_size=32, shuffle=True, num_workers=12)
dataloader_val = DataLoader(cns_val, batch_size=32, shuffle=True, num_workers=12)


# TODO - tensorboard; generalizing vs. memorizing algo; dynamically add layers while preserving previous weight values

class M(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_neurons_per_layer = 16

        self.input_layer = nn.Linear(4, self.num_neurons_per_layer, bias=True).to(device)
        self.output_layer = nn.Linear(self.num_neurons_per_layer, 4, bias=True).to(device)  # linear output

        self.num_hidden_layers = 1  # default, start small

        self.h = []  # will hold all hidden layers
        for _ in range(self.num_hidden_layers):
            self.h.append(
                self.BaseLayer(self.num_neurons_per_layer, self.num_neurons_per_layer).to(device)
            )

    def forward(self, x):
        x = F.relu6(self.input_layer(x))
        for h in self.h:
            x = F.relu6(h(x))
        x = self.output_layer(x)  # keep linear, do not apply activation function

        return x

    def BaseLayer(self, in_features, out_features):
        return nn.Linear(in_features, out_features, bias=True)


model = M()
model = model.to(device)

optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False
)

loss_function = nn.MSELoss(reduction='sum')  # 'none' | 'mean' | 'sum'

epochs = 1000
iter_num_ctr = 0
for t in range(epochs):  # epochs
    # put model in train mode
    model = model.train()
    losses_train = []
    for batch_idx, batch in enumerate(dataloader_train):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()  # clear param gradients

        y_pred = model(x)

        loss_train = loss_function(y_pred, y)
        loss_train.backward()
        optimizer.step()

        losses_train.append(loss_train.cpu().item())


    # put model in evaluation mode (no updates to network)
    model = model.eval()
    losses_val = []
    for batch_idx, batch in enumerate(dataloader_val):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        loss_val = loss_function(y_pred, y)

        losses_val.append(loss_val.cpu().item())

    # End of epoch reporting
    print(f"Epoch {t}/{epochs}")
    print(f"Loss (MSE)\t\tTrain\t{np.mean(losses_train)}\t\tVal\t{np.mean(losses_val)}")
    print()
