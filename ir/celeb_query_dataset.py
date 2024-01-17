import os
from typing import List

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

annotation_csv_path = '../celebA_ir/celebA_anno_query.csv'
query_photos_path = '../celebA_ir/celebA_query'


class CelebQueryDataset(Dataset):
    def __init__(self):
        self.annotations = pd.read_csv(annotation_csv_path)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        image_name = row['img']
        person_id = row['id']
        img = Image.open(os.path.join(query_photos_path, image_name))
        transform = transforms.Compose([transforms.ToTensor()])
        tensor_image = transform(img)
        return tensor_image, person_id

    def get_annotations(self) -> pd.DataFrame:
        return self.annotations

    def get_person_ids(self) -> List[int]:
        return self.annotations['id'].unique().tolist()

    def get_image_indices(self,  person_id: int) -> List[int]:
        annotations = self.get_annotations()
        return annotations.index[annotations['id'] == person_id].tolist()