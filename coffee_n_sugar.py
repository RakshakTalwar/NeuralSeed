from pkg_resources import resource_filename

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from torch.utils.data import Dataset

import math


class CoffeeNSugar(Dataset):

    def __init__(self, split: str, train_fraction=0.7):
        """
        Train has 70% of total records by default
        Val has remaining 30% of records

        :param split: must be either "train" or "val"
        :param train_fraction: number in range (0., 1.)
        """

        super().__init__()

        self.train_fraction = train_fraction

        split_acceptable_values = ('train', 'val')
        assert split in split_acceptable_values, f"Acceptable values for split are: {split_acceptable_values}"

        self.split = split

        self.coffee_df = pd.read_csv(
            resource_filename("data", "coffee-prices-historical-chart-data.csv"), sep=",", header=0,
            names=['date', 'value'],
            skiprows=2687)
        self.coffee_df['value_normalized'] =\
            Normalizer(norm='max').fit_transform(self.coffee_df['value'].values.reshape((1, -1))).ravel()


        self.sugar_df = pd.read_csv(
            resource_filename("data", "sugar-prices-historical-chart-data.csv"), sep=",", header=0,
            names=['date', 'value'],
            skiprows=13)
        self.sugar_df['value_normalized'] =\
            Normalizer(norm='max').fit_transform(self.sugar_df['value'].values.reshape((1, -1))).ravel()

        if len(self.coffee_df) != len(self.sugar_df) or len(self.coffee_df) == 0:
            raise ValueError("Both dataframes should have datapoints and the same number of datapoints as each other")
        n_records_total = len(self.coffee_df)
        self.n_records_train = int(self.train_fraction * n_records_total)
        self.n_records_val = n_records_total - self.n_records_train

        self.sugar_df_train, self.sugar_df_val = self.sugar_df.iloc[:self.n_records_train, :],\
                                                 self.sugar_df.iloc[self.n_records_train:, :]

        self.coffee_df_train, self.coffee_df_val = self.coffee_df.iloc[:self.n_records_train, :],\
                                                   self.coffee_df.iloc[self.n_records_train:, :]

        self.pane_size = 4 # how many sequential tokens there are per side of neural net (#output units = #input units)
        """
          - - - -      # output (pane), sequence with x-axis being time--->
        - - - -        # input  (pane), sequence with x-axis being time--->
        0 1 2 3 4      # indices for input and output above
        """
        self.frame_size = self.pane_size + 1

    def get_highest_valid_index(self):
        n_points = self.n_records_train if self.split == "train" else self.n_records_val
        highest_valid_idx = n_points - (self.pane_size + (self.frame_size - self.pane_size))
        return highest_valid_idx

    def __len__(self):
        return self.get_highest_valid_index() + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sugar is input; coffee output
        input_indices = np.arange(idx, idx+self.pane_size)
        output_indices = input_indices + (self.frame_size - self.pane_size)

        if self.split == "train":
            input_sequence = self.sugar_df_train['value_normalized'].iloc[input_indices].values
            output_sequence = self.coffee_df_train['value_normalized'].iloc[output_indices].values
        elif self.split == "val":
            input_sequence = self.sugar_df_val['value_normalized'].iloc[input_indices].values
            output_sequence = self.coffee_df_val['value_normalized'].iloc[output_indices].values
        else:
            raise ValueError("This instance's split parameter was not correctly set")

        return \
            torch.as_tensor(input_sequence, dtype=torch.float32), torch.as_tensor(output_sequence, dtype=torch.float32)

if __name__ == "__main__":
    cns = CoffeeNSugar('train')
