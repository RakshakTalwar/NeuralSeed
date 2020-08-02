from pkg_resources import resource_filename

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from torch.utils.data import Dataset


class CoffeeNSugar(Dataset):

    def __init__(self):
        super().__init__()

        self.coffee_df = pd.read_csv(
            resource_filename("data", "coffee-prices-historical-chart-data.csv"), sep=",", header=0,
            names=['date', 'value'],
            skiprows=2687)
        self.coffee_df['value_normalized'] = Normalizer(norm='max').fit_transform(self.coffee_df['value'].values.reshape((1, -1))).ravel()


        self.sugar_df = pd.read_csv(
            resource_filename("data", "sugar-prices-historical-chart-data.csv"), sep=",", header=0,
            names=['date', 'value'],
            skiprows=13)
        self.sugar_df['value_normalized'] = Normalizer(norm='max').fit_transform(self.sugar_df['value'].values.reshape((1, -1))).ravel()

        self.pane_size = 4 # how many sequential tokens there are per side of neural net (#output units = #input units)
        """
          - - - -      # output (pane), sequence with x-axis being time--->
        - - - -        # input  (pane), sequence with x-axis being time--->
        0 1 2 3 4      # indices for input and output above
        """
        self.frame_size = self.pane_size + 1

    def __len__(self):
        if len(self.coffee_df) == len(self.sugar_df):
            return len(self.coffee_df)
        else:
            raise ValueError("Both dataframes should have the same number of datapoints")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > (len(self) - (self.pane_size+1)):  # invalid idx requested
            return self.__getitem__(idx=np.random.randint(low=0, high=len(self)))  # try again with a random idx

        # sugar is input; coffee output
        input_indices = np.arange(idx, idx+self.pane_size)
        output_indices = input_indices + (self.frame_size - self.pane_size)

        input_sequence = self.sugar_df['value_normalized'].iloc[input_indices].values
        output_sequence = self.coffee_df['value_normalized'].iloc[output_indices].values

        return \
            torch.as_tensor(input_sequence, dtype=torch.float32), torch.as_tensor(output_sequence, dtype=torch.float32)
