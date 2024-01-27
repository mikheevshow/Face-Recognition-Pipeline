from torch import Tensor
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import pandas as pd
import numpy as np

available_modes = ['train', 'val', 'test']
availaible_file_locations = ['Kaggle', 'Local']

kaggle_root_data_path = '/kaggle/input/celeba-train-500/celebA_train_500/'
kaggle_images_path = '/kaggle/input/celeba-train-500/celebA_train_500/celebA_imgs/'


local_root_data_path = './celebA_train_500/'
local_images_path = './celebA_train_500/celebA_imgs/'


def get_image_path(env: str = 'Kaggle') -> str:
    return local_root_data_path

import random


class CelebTripletDataset(Dataset):
    """
    Celeb 500 classes dataset for triplet loss learning
    """
    def __init__(self, mode: str = 'train', file_location: str = 'Local'):
        if mode not in available_modes:
            raise ValueError(f'Unrecognized mode: {mode}')
        self._mode = mode
        self._file_location = get_image_path(file_location)
        self._images_path = self._file_location + 'celebA_imgs/'
        self._train_val_test_split_df = pd.read_csv(
            filepath_or_buffer=f'{self._file_location}celebA_train_split.txt',
            sep=' ',
            names=['Image', 'Mode']
        )
        self._image_annos = pd.read_csv(
            f'{self._file_location}celebA_anno.txt',
            sep=' ',
            names=['Image', 'Person']
        )
        self._df = self.get_data_depend_on_mode()

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, item) -> (Tensor, Tensor, Tensor):

        df = self._df

        anchor_image_name = df.iloc[item]['Image']
        anchor_label = df.iloc[item]['Person']

        positive_list = df[(df['Person'] == anchor_label) & (df['Image'] != anchor_image_name)]['Image'].tolist()
        positive_image_name = random.choice(positive_list)

        negative_list = df[df['Person'] != anchor_label]['Image'].tolist()
        negative_image_name = random.choice(negative_list)

        anchor_path = self._images_path + anchor_image_name
        pos_path = self._images_path + positive_image_name
        neg_path = self._images_path + negative_image_name

        anchor_img = cv2.imread(anchor_path)
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)[77:-41, 45:-50]

        pos_img = cv2.imread(pos_path)
        pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2RGB)[77:-41, 45:-50]

        neg_img = cv2.imread(neg_path)
        neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)[77:-41, 45:-50]

        transforms = self.get_transforms()

        transformed_anchor_img = transforms(image=anchor_img)['image']
        transformed_pos_img = transforms(image=pos_img)['image']
        transformed_neg_img = transforms(image=neg_img)['image']

        return transformed_anchor_img, transformed_pos_img, transformed_neg_img

    def get_transforms(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self._mode == 'train':
            return A.Compose([
                #A.Normalize(mean, std),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(mean, std),
                ToTensorV2()
            ])

    def get_data_depend_on_mode(self) -> pd.DataFrame:
        images: np.ndarray = None
        df = self._train_val_test_split_df
        if self._mode == 'train':
            images = df[df['Mode'] == 0]['Image'].to_numpy(dtype=str)
        elif self._mode == 'val':
            images = df[df['Mode'] == 1]['Image'].to_numpy(dtype=str)
        elif self._mode == 'test':
            images = df[df['Mode'] == 2]['Image'].to_numpy(dtype=str)
        else:
            raise ValueError(f'Unrecognized mode: {self._mode}')
        image_annos = self._image_annos
        return image_annos[image_annos['Image'].isin(images)]

