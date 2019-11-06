from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
from torchvision import transforms


class DVTDataset(Dataset):
    """DVT dataset."""

    def __init__(self, csv_file, root_dir, device, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.annotation_map = pd.read_csv(csv_file)
        p = transforms.Compose([transforms.ToPILImage(), transforms.Scale((224, 224)), transforms.ToTensor()])

        self.root_dir = root_dir
        self.transform = p

    def __len__(self):
        return len(self.annotation_map)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = self.annotation_map.loc[idx, "reference_image_filename"]
        img_path = str(self.annotation_map.loc[idx, "reference_image_filename"])
        image = pydicom.dcmread(img_path)
        img_array = image.pixel_array

        # print(img_array[1:10, 1:10])
        img_array = img_array - np.mean(img_array)
        img_array = img_array / np.std(img_array)
        # img_array = self.preprocess_image(img_array)
        # plt.imshow(image.pixel_array)
        access_number = self.annotation_map.loc[idx, "accession_number"]
        name = self.annotation_map.loc[idx, "patient_name"]
        clean_label = self.annotation_map.loc[idx, "clean_label"]
        slice_num = self.annotation_map.loc[idx, "slice_number_in_seg_file"]

        img_array = np.reshape(img_array, (img_array.shape[0], img_array.shape[1], 1))

        # img_array = torch.reshape(torch.tensor(img_array), (1, img_array.shape[0], img_array.shape[1]))
        # img_array = np.reshape(img_array, (1, img_array.shape[0], img_array.shape[1]))
        # print(img_array.shape)
        img_array = self.transform(np.float32(img_array)).numpy()

        sample = {'image': img_array, 'access_number': access_number, 'label': clean_label, 'slice_num': slice_num}

        return sample

    def preprocess_image(self, image):
        image -= self.meanThe
        image /= self.std
        return image