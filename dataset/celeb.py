from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import pandas as pd
import numpy as np

from commons import available_modes, get_image_path


class CelebDataset(Dataset):
    def __init__(self, mode: str = 'train', file_location: str = 'Kaggle'):
        if mode not in available_modes:
            raise ValueError(f'Unrecognized mode: {mode}')
        self._file_location = file_location
        df = pd.read_csv('/kaggle/input/celeba-train-500/celebA_train_500/celebA_train_split.txt', sep=' ', names=['Image', 'Mode'])
        self.image_annos = pd.read_csv('/kaggle/input/celeba-train-500/celebA_train_500/celebA_anno.txt', sep=' ', names=['Image', 'Person'])
        self.train_images = df[df['Mode'] == 0]['Image'].to_numpy(dtype=str)
        self.val_images = df[df['Mode'] == 1]['Image'].to_numpy(dtype=str)
        self.test_images = df[df['Mode'] == 2]['Image'].to_numpy(dtype=str)
        self._mode = mode

    def __len__(self):
        return len(self.get_images_depend_on_mode())

    def __getitem__(self, index):
        image_name = self.get_images_depend_on_mode()[index]
        image_path = get_image_path(self._file_location) + 'celebA_imgs/' + image_name

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[77:-41, 45:-50]

        transform = self.get_transforms()
        transformed_image = transform(image=image)["image"]
        annotation: pd.DataFrame = self.image_annos
        person_id = annotation[annotation['Image'] == image_name]['Person'].values[0]
        return transformed_image, person_id

    def get_transforms(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self._mode == 'train':
            return A.Compose([
                A.Cutout(),
                A.Normalize(mean, std),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(mean, std),
                ToTensorV2()
            ])

    def get_images_depend_on_mode(self) -> np.array:
        if self._mode == 'train':
            return self.train_images
        elif self._mode == 'val':
            return self.val_images
        elif self._mode == 'test':
            return self.test_images
        else:
            raise ValueError(f'Unrecognized mode: {self._mode}')

