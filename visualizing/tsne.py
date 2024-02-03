import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import numpy as np

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import random


def tsne_visualization(
        model: Module,
        dataloader: DataLoader,
        class_count: int,
        dataloader_iterations: int = 5,
        visualize_random_classes_amount: int = 10,
        device=torch.device("cpu")):
    model.eval()

    _labels = None
    _outputs = None

    for i, (images, labels) in enumerate(dataloader):
        if i > dataloader_iterations:
            break

        with torch.inference_mode():
            images = images.to(device)
            outputs = model.get_embedding(images)

            labels = labels.cpu().numpy()
            if _labels is None:
                _labels = labels
            else:
                _labels = np.concatenate((_labels, labels), axis=0)

            outputs = outputs.cpu().numpy()
            if _outputs is None:
                _outputs = outputs
            else:
                _outputs = np.concatenate((_outputs, outputs), axis=0)

    tsne = TSNE(n_components=2).fit_transform(_outputs)

    def scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors_per_class = {}

    for i in range(visualize_random_classes_amount):
        random_class = random.randint(1, class_count)
        if random_class not in colors_per_class:
            color = []
            for j in range(3):
                color.append(round(random.uniform(0.0, 255.0), 2))
            colors_per_class[random_class] = color

    for label in colors_per_class:
        indices = [i for i, l in enumerate(_labels) if l == label]

        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        color = np.array(colors_per_class[label], dtype=np.float) / 255

        ax.scatter(current_tx, current_ty, c=color, edgecolors=['black'], label=label)

    plt.show()
