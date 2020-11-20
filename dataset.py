import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from skimage import io, transform
import numpy as np
import asf_read as reader


class FaceLandmarksDataset(Dataset):
    def __init__(self,root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # this should return a list of all landmarks from all images
        # or maybe a dictionary of names and dfs
        self.landmarks_dict = reader.all_landmarks(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_dict)

    # support indexing such that dataset[i] can be used to get the ith sample
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        keys_list = list(self.landmarks_dict)
        name = keys_list[idx]

        # image is colored because we get it directly from directory
        img_name = self.root_dir + name + ".jpg"
        #print(img_name)

        image = io.imread(img_name)
        landmarks = self.landmarks_dict[name]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class NosetipDataset(Dataset):
    def __init__(self,root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # this should return a list of all landmarks from all images
        # or maybe a dictionary of names and dfs
        self.nose_dict = reader.all_noses(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.nose_dict)

    # support indexing such that dataset[i] can be used to get the ith sample
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        keys_list = list(self.nose_dict)
        name = keys_list[idx]

        # image is colored because we get it directly from directory
        img_name = self.root_dir + name + ".jpg"
        #print(img_name)

        image = io.imread(img_name)
        landmarks = self.nose_dict[name]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample