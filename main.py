from typing import Optional, Union, Sequence, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coffee_n_sugar import CoffeeNSugar


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cns = CoffeeNSugar()
dataloader = DataLoader(cns, batch_size=32, shuffle=True, num_workers=12)


# TODO - function on GPU; validation split; tensorboard

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

# def training_step(self, batch, batch_idx):
#
#
#
#     tensorboard_logs = {'train_loss': loss}
#     return {'loss': loss, 'log': tensorboard_logs}

model = M()
model = model.to(device)

optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False
)

loss_function = nn.MSELoss(reduction='sum')  # 'none' | 'mean' | 'sum'

epochs = 10
iter_num_ctr = 0
for t in range(epochs):  # epochs
    for batch_idx, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter_num_ctr % 100 == 0:
            print(f"Epoch {t}/{epochs} Iteration {batch_idx}")
            print(f"MSE\t{loss.cpu().item()}")
            print()
        iter_num_ctr += 1



