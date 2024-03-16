##################################
# Clay Crews, Lex Whalen
#
# Dataset class for Brain MRI
# For use in later pipelines.

# Necessary imports
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from glob import glob
import cv2
import numpy as np

class BrainMRIDatasetBuilder:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def create_df(self):
        images_paths = []
        masks_paths = glob(f'{self.data_dir}/*/*_mask*')

        for i in masks_paths:
            images_paths.append(i.replace('_mask', ''))

        df = pd.DataFrame(data={'images_paths': images_paths, 'masks_paths': masks_paths})

        return df

    def split_df(self, df):
        train_df, dummy_df = train_test_split(df, train_size=0.8)
        valid_df, test_df = train_test_split(dummy_df, train_size=0.5)
        return train_df, valid_df, test_df


class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, mask_transform=None):
        self.df = dataframe
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0]) / 255.0
        mask = cv2.imread(self.df.iloc[idx, 1]) / 255.0
        mask = np.where(mask >= 0.5, 1., 0.)
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask