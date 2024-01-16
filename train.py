import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from tqdm.notebook import trange


def accuracy(logits: torch.Tensor, ground_truth_labels: torch.Tensor) -> float:
    predicted_labels = torch.argmax(logits, dim=1)
    num_of_correct_predictions = torch.sum(predicted_labels == ground_truth_labels)
    return num_of_correct_predictions.item()


def train_step(model: Module, train_dataloader: DataLoader, criterion, optimizer: Optimizer, device) -> (float, float):
    """

    :param model:
    :param train_dataloader:
    :return: Tuple[float, float]: train_loss, train_accuracy
    """
    model.train(mode=True)
    train_loss, train_acc = 0.0, 0.0
    for _, (images, ground_truth_labels) in enumerate(train_dataloader):
        images: Tensor = images.to(device=device)
        ground_truth_labels = ground_truth_labels.to(device=device)
        output = model(images)
        loss: Tensor = criterion(output, ground_truth_labels)
        train_loss += loss.item()
        train_acc += accuracy(output, ground_truth_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_dataloader), 100 * (train_acc / len(train_dataloader))


def validation_step(model: Module, val_dataloader: DataLoader, criterion, device) -> (float, float):
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.inference_mode():
        for _, (images, labels) in enumerate(val_dataloader):
            images: Tensor = images.to(device=device)
            labels = labels.to(device=device)
            output = model(images)
            loss: Tensor = criterion(output, labels)
            val_loss += loss.item()
            val_acc += accuracy(output, labels)
    return val_loss / len(val_dataloader), 100.0 * (val_acc / len(val_dataloader))


def train(
        epochs: int,
        model: Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion,
        optimizer: Optimizer,
        device) -> (float, float):

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    summary_writer = SummaryWriter(f'runs/face_recognition_trainer_{timestamp}')

    for epoch in trange(epochs):

        train_loss, train_acc = train_step(
            model=model,
            train_dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = validation_step(
            model=model,
            val_dataloader=val_dataloader,
            criterion=criterion,
            device=device
        )

        print(
            f'Epoch: {epoch+1} | '
            f'Train loss {train_loss:.4f} | '
            f'Train accuracy {train_acc:.4f} | '
            f'Validation loss {val_loss:.4f} | '
            f'Validation accuracy {val_acc:.4f}'
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

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

        summary_writer.add_graph(
            model=model,
            input_to_model=torch.randn(32, 3, 100, 83).to(device)
        )

    summary_writer.close()
    print(f'Finished training.')
    return history


