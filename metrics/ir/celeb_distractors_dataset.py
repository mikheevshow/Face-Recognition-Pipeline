import os

from torch.utils.data import Dataset

from PIL import Image

from torchvision.transforms import transforms

distractors_photos_path = '../../celebA_ir/celebA_distractors'


class CelebDistractorsDataset(Dataset):
    def __init__(self):
        super(CelebDistractorsDataset, self).__init__()
        self.img_names = os.listdir(distractors_photos_path)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = Image.open(os.path.join(distractors_photos_path, img_name))
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        return img

