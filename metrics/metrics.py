import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


def accuracy(model: Module, dataloader: DataLoader, criterion, device=None) -> (float, float):
    if device is None:
        device = torch.device("cpu")
    model.eval()
    total_loss = 0.0
    correct_answers = 0
    total_labels = 0
    with torch.inference_mode():
        for _, (images, labels) in enumerate(dataloader):
            images: Tensor = images.to(device=device)
            labels = labels.to(device=device)
            output = model(images)
            loss: Tensor = criterion(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_labels += labels.size(0)
            correct_answers += (predicted == labels).sum().item()
    return total_loss / len(dataloader), 100.0 * correct_answers // total_labels

