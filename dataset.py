from torch.utils.data import Dataset
from torchvision import transforms as tf

from PIL import Image

import pandas as pd
import numpy as np


available_modes = ['train', 'val', 'test']
images_path = './celebA_train_500/celebA_imgs/'


class CelebDataset(Dataset):
    def __init__(self, mode: str = 'train'):
        if mode not in available_modes:
            raise ValueError(f'Unrecognized mode: {mode}')
        df = pd.read_csv('./celebA_train_500/celebA_train_split.txt', sep=' ', names=['Image', 'Mode'])
        self.image_annos = pd.read_csv('./celebA_train_500/celebA_anno.txt', sep=' ', names=['Image', 'Person'])
        self.train_images = df[df['Mode'] == 0]['Image'].to_numpy(dtype=str)
        self.val_images = df[df['Mode'] == 1]['Image'].to_numpy(dtype=str)
        self.test_images = df[df['Mode'] == 2]['Image'].to_numpy(dtype=str)
        self._mode = mode

    def __len__(self):
        return len(self.get_images_depend_on_mode())

    def __getitem__(self, index):
        image_name = self.get_images_depend_on_mode()[index]
        image_path = images_path + image_name
        image = np.array(Image.open(image_path))
        image = Image.fromarray(image[77:-41, 45:-50])
        transform = tf.Compose([tf.ToTensor()])
        transformed_image = transform(image)
        annotation: pd.DataFrame = self.image_annos
        person_id = annotation[annotation['Image'] == image_name]['Person'].values[0]
        return transformed_image, person_id

    def get_images_depend_on_mode(self) -> np.array:
        if self._mode == 'train':
            return self.train_images
        elif self._mode == 'val':
            return self.val_images
        elif self._mode == 'test':
            return self.test_images
        else:
            raise ValueError(f'Unrecognized mode: {self._mode}')

