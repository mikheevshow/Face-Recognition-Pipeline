import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import numpy as np

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import random


def tsne(model: Module, dataloader: DataLoader, device):
    model.eval()
    _labels = None
    _outputs = None
    for i, (images, labels) in enumerate(dataloader):
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
    classes = np.unique(_labels)
    class_to_color = {}
    for i in classes:
        color = []
        for j in range(3):
            color.append(round(random.uniform(0.0, 255.0), 2))
        class_to_color[i] = color

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

    for label in class_to_color:
        if label in range(0, 500):
            indices = [i for i, l in enumerate(_labels) if l == label]

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            color = np.array(class_to_color[label], dtype=np.float64) / 255
            ax.scatter(current_tx, current_ty, c=color, edgecolors=['black'], label=label)

            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.suptitle(f'TSNE for {512}d embeddings', fontsize=16)
    ax.set_xlabel('TSNE1')
    ax.set_ylabel('TSNE2')
    plt.show()

