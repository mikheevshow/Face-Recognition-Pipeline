import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss.triplet_loss import triplet_loss

from datetime import datetime

from tqdm.notebook import trange


def train_step(model: Module, train_dataloader: DataLoader, criterion, optimizer: Optimizer, device) -> float:
    model.train(mode=True)
    train_loss = 0.0
    for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
        optimizer.zero_grad()

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        loss: Tensor = criterion(anchor_embedding, positive_embedding, negative_embedding)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    return train_loss / len(train_dataloader)


def validation_step(model: Module, val_dataloader: DataLoader, criterion, device) -> (float, float):
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for batch_idx, (anchor, positive, negative) in enumerate(val_dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss: Tensor = criterion(anchor_embedding, positive_embedding, negative_embedding)
            val_loss += loss.item()

    return val_loss / len(val_dataloader)


def train(epochs: int,
          model: Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          optimizer: Optimizer,
          device):

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    summary_writer = SummaryWriter(f'runs/face_recognition_trainer_{timestamp}')

    criterion = triplet_loss

    for epoch in trange(epochs):

        train_loss = train_step(model, train_dataloader, criterion, optimizer, device)
        val_loss = validation_step(model, val_dataloader, criterion, device)

        print(
            f'Epoch: {epoch+1} | '
            f'Train loss {train_loss:.4f} | '
            f'Validation loss {val_loss:.4f} | '
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        summary_writer.add_scalars(
            main_tag='Loss',
            tag_scalar_dict={'train_loss': train_loss, 'val_loss': val_loss},
            global_step=epoch
        )

    summary_writer.close()
    print(f'Finished training.')
    return history

