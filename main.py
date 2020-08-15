from typing import Optional, Union, Sequence, Dict, Tuple, List

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from coffee_n_sugar import CoffeeNSugar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO - CLI options for hyperparams; generalizing vs. memorizing algo; dynamically add layers while preserving previous weight values
# OPTIONAL TODO, use https://neptune.ai/ to track experiments. Plugs into PyTorch Lightning easily

class M(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.num_neurons_per_layer = 16

        self.input_layer = nn.Linear(4, self.num_neurons_per_layer, bias=True).to(device)
        self.output_layer = nn.Linear(self.num_neurons_per_layer, 4, bias=True).to(device)  # linear output

        self.num_hidden_layers = 1  # default, start small

        self.loss_function = nn.MSELoss(reduction='sum')  # 'none' | 'mean' | 'sum'

        self.h = []  # will hold all hidden layers
        for _ in range(self.num_hidden_layers):
            self.h.append(
                self.BaseLayer(self.num_neurons_per_layer, self.num_neurons_per_layer).to(device)
            )

    def BaseLayer(self, in_features, out_features):
        return nn.Linear(in_features, out_features, bias=True)
    
    def train_dataloader(self):
        cns_train = CoffeeNSugar(split='train', train_fraction=.7)
        return DataLoader(cns_train, batch_size=32, shuffle=True, num_workers=12)

    def val_dataloader(self):
        cns_val = CoffeeNSugar(split='val', train_fraction=.7)
        return DataLoader(cns_val, batch_size=32, shuffle=True, num_workers=12)

    def forward(self, x):
        x = F.relu6(self.input_layer(x))
        for h in self.h:
            x = F.relu6(h(x))
        x = self.output_layer(x)  # keep linear, do not apply activation function

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss,
                'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss,
                'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=1e-3,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)

epochs = 1000

model = M()
trainer = pl.Trainer(
    progress_bar_refresh_rate=20,
    profiler=True,
    max_epochs=epochs,
    gpus=1,
    precision=32
)
trainer.fit(model)
