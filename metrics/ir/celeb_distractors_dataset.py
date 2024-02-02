import os

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

distractors_photos_path = '../../celebA_ir/celebA_distractors'


class CelebDistractorsDataset(Dataset):
    def __init__(self):
        super(CelebDistractorsDataset, self).__init__()
        self.img_names = os.listdir(distractors_photos_path)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = cv2.imread(os.path.join(distractors_photos_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = self.get_transforms()
        img = transform(image=img)['image']
        return img

    def get_transforms(self):
        resize_height, resize_width = 224, 224
        return A.Compose([
            A.Resize(resize_height, resize_width),
            A.Normalize(),
            ToTensorV2()
        ])
