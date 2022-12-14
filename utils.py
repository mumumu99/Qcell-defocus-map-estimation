import torch
import torchvision
from dataset import QcellTrainDataset, QcellValDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    val_dir,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_ds = QcellTrainDataset(
        image_dir=train_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    val_ds = QcellValDataset(
        image_dir=val_dir,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader