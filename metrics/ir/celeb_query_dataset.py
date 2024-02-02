import os
from typing import List

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd

annotation_csv_path = '../../celebA_ir/celebA_anno_query.csv'
query_photos_path = '../../celebA_ir/celebA_query'


class CelebQueryDataset(Dataset):
    def __init__(self):
        self.annotations = pd.read_csv(annotation_csv_path)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        image_name = row['img']
        person_id = row['id']
        img = cv2.imread(os.path.join(query_photos_path, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = self.get_transforms()
        tensor_image = transform(image=img)['image']
        return tensor_image, person_id

    def get_transforms(self):
        resize_height, resize_width = 224, 224
        return A.Compose([
            A.Resize(resize_height, resize_width),
            A.Normalize(),
            ToTensorV2()
        ])

    def get_annotations(self) -> pd.DataFrame:
        return self.annotations

    def get_person_ids(self) -> List[int]:
        return self.annotations['id'].unique().tolist()

    def get_image_indices(self,  person_id: int) -> List[int]:
        annotations = self.get_annotations()
        return annotations.index[annotations['id'] == person_id].tolist()

