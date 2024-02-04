from torch.utils.data import Dataset
from torchvision import transforms as tf
from matplotlib import pyplot as plt

import random


def draw_samples(dataset: Dataset):
    nrows, ncols = 1,  10
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 7))
    ax = ax.flatten()
    to_pil = tf.ToPILImage()
    for i in range(nrows * ncols):
        rand_index = random.randint(0, len(dataset) - 1)
        image_tensor = dataset[rand_index][0]
        pil_image = to_pil(image_tensor)
        ax[i].imshow(pil_image)