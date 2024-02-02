from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import pandas as pd
import numpy as np

available_modes = ['train', 'val', 'test']
available_file_locations = ['kaggle', 'local']

kaggle_root_data_path = '/kaggle/input/celeba-train-500/celebA_train_500/'
local_root_data_path = './celebA_train_500/'


def get_image_path(env: str = 'local') -> str:
    env = env.lower()
    if env not in available_file_locations:
        raise ValueError(f'Unsupported environment {env}')
    if env == 'kaggle':
        return kaggle_root_data_path
    else:
        return local_root_data_path


class CelebDataset(Dataset):
    def __init__(self, mode: str = 'train', env: str = 'local'):
        if mode not in available_modes:
            raise ValueError(f'Unrecognized mode: {mode}')
        self._file_location = get_image_path(env)
        self._images_path = self._file_location + 'celebA_imgs/'
        train_val_split = pd.read_csv(
            f'{self._file_location}celebA_train_split.txt',
            sep=' ',
            names=['Image', 'Mode'])

        image_annos = pd.read_csv(
            f'{self._file_location}celebA_anno.txt',
            sep=' ',
            names=['Image', 'Target'])

        (train_split_class_annotations,
         val_split_class_annotations,
         test_split_class_annotations) = self.prepare_data(train_val_split, image_annos)

        self.train_images = train_split_class_annotations
        self.val_images = val_split_class_annotations
        self.test_images = test_split_class_annotations
        self._mode = mode

    def __len__(self):
        return len(self.get_images_depend_on_mode())

    def __getitem__(self, index):
        data = self.get_images_depend_on_mode()
        image_name = data.iloc[index]['Image']
        image_path = self._images_path + image_name

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[77:-41, 45:-50]

        transform = self.get_transforms()
        transformed_image = transform(image=image)["image"]

        target = data.iloc[index]['Target']

        return transformed_image, target

    def get_transforms(self):

        resize_train_height, resize_train_width = 224, 224
        resize_val_height, resize_val_width = 224, 224

        if self._mode == 'train':
            return A.Compose([
                # Add augmentations here
                A.Resize(height=resize_train_height, width=resize_train_width),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=resize_val_height, width=resize_val_width),
                A.Normalize(),
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

    def prepare_data(self, train_val_split: pd.DataFrame, class_annotations: pd.DataFrame):

        train_split = train_val_split[train_val_split['Mode'] == 0]
        val_split = train_val_split[train_val_split['Mode'] == 1]
        test_split = train_val_split[train_val_split['Mode'] == 2]

        train_split_image_names = train_split['Image'].tolist()
        val_split_image_names = val_split['Image'].tolist()
        test_split_image_names = test_split['Image'].tolist()

        train_split_class_annotations = class_annotations[class_annotations['Image'].isin(train_split_image_names)]
        val_split_class_annotations = class_annotations[class_annotations['Image'].isin(val_split_image_names)]
        test_split_class_annotations = class_annotations[class_annotations['Image'].isin(test_split_image_names)]

        train_split_class_counts = train_split_class_annotations['Target'].value_counts()

        valid_classes = train_split_class_counts[train_split_class_counts >= 10].index.tolist()
        valid_classes = sorted(valid_classes)
        valid_classes_mapping = {old_class: new_class for new_class, old_class in enumerate(valid_classes)}

        non_valid_classes = train_split_class_counts[train_split_class_counts < 10].index.tolist()
        non_valid_classes = sorted(non_valid_classes)

        train_split_class_annotations = train_split_class_annotations[
            train_split_class_annotations['Target'].isin(valid_classes)]
        val_split_class_annotations = val_split_class_annotations[
            val_split_class_annotations['Target'].isin(valid_classes)]
        test_split_class_annotations = test_split_class_annotations[
            test_split_class_annotations['Target'].isin(valid_classes)]

        train_split_class_annotations['Target'] = train_split_class_annotations['Target'].map(valid_classes_mapping)
        val_split_class_annotations['Target'] = val_split_class_annotations['Target'].map(valid_classes_mapping)
        test_split_class_annotations['Target'] = test_split_class_annotations['Target'].map(valid_classes_mapping)

        train_split_class_annotations = train_split_class_annotations.reset_index()
        val_split_class_annotations = val_split_class_annotations.reset_index()
        test_split_class_annotations = test_split_class_annotations.reset_index()

        return train_split_class_annotations, val_split_class_annotations, test_split_class_annotations

    def class_count(self):
        return len(self.train_images['Target'].unique())
