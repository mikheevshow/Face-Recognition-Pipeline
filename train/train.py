from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from tqdm.notebook import trange


def train_step(
        model: Module,
        train_dataloader: DataLoader,
        criterion,
        optimizer: Optimizer,
        device,
        train_arc_face: bool) -> Tuple[float, float]:
    """
    :param criterion: loss function
    :param model: model
    :param train_dataloader:
    :return: Tuple[float, float]: train_loss, train_accuracy
    """
    model.train(mode=True)
    train_loss = 0.0
    correct_answers = 0
    total_labels = 0
    for _, (images, labels) in enumerate(train_dataloader):
        images: Tensor = images.to(device=device)
        labels = labels.to(device=device)
        if train_arc_face:
            output = model(images, labels)
        else:
            output = model(images)
        loss: Tensor = criterion(output, labels)
        train_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total_labels += labels.size(0)
        correct_answers += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_dataloader), 100 * correct_answers / total_labels


def validation_step(
        model: Module,
        val_dataloader: DataLoader,
        criterion,
        device, train_arc_face: bool) -> Tuple[float, float]:

    model.eval()
    val_loss = 0.0
    correct_answers = 0
    total_labels = 0
    with torch.inference_mode():
        for _, (images, labels) in enumerate(val_dataloader):
            images: Tensor = images.to(device=device)
            labels = labels.to(device=device)
            if train_arc_face:
                output = model(images, labels)
            else:
                output = model(images)
            loss: Tensor = criterion(output, labels)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_labels += labels.size(0)
            correct_answers += (predicted == labels).sum().item()
    return val_loss / len(val_dataloader), 100.0 * correct_answers / total_labels


def train(
        epochs: int,
        model: Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion,
        optimizer: Optimizer,
        learning_rate_scheduler=None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        train_arc_face: bool = False):
    torch.cuda.empty_cache()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    summary_writer = SummaryWriter(f'runs/face_recognition_trainer_{timestamp}')

    for epoch in trange(epochs):

        model.to(device)

        train_loss, train_acc = train_step(
            model=model,
            train_dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_arc_face=train_arc_face
        )

        val_loss, val_acc = validation_step(
            model=model,
            val_dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            train_arc_face=train_arc_face
        )

        before_lr = optimizer.param_groups[0]["lr"]
        if learning_rate_scheduler is not None:
            learning_rate_scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]

        print(
            f'Epoch: {epoch + 1} | '
            f'Train loss {train_loss:.4f} | '
            f'Train accuracy {train_acc:.4f} | '
            f'Validation loss {val_loss:.4f} | '
            f'Validation accuracy {val_acc:.4f} | '
            f'Learning rate {before_lr:.4f} -> {after_lr:.4f} | '
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(before_lr)

        summary_writer.add_scalars(
            main_tag='Loss',
            tag_scalar_dict={'train_loss': train_loss, 'val_loss': val_loss},
            global_step=epoch
        )

        summary_writer.add_scalars(
            main_tag='Accuracy',
            tag_scalar_dict={'train_acc': train_acc, 'val_acc': val_acc},
            global_step=epoch
        )

        summary_writer.add_scalars(
            main_tag='Learning Rate',
            tag_scalar_dict={'lr': before_lr},
            global_step=epoch
        )

    torch.cuda.empty_cache()
    summary_writer.close()
    print(f'Finished training.')
    return history
