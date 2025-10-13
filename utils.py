from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch

device = torch.device("cpu")
DATA_ROOT = "./data"

# 数据增强配置
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])
# -------------------------------------------------------------------------------------

# 获取CIFAR数据集的DataLoader
def get_dataloaders(batch_size: int, DEVICE):
    if not torch.cuda.is_available():
        DATA_ROOT = "D:\\SHIVAAAA\\Documents\\MinGW\\NeuralNetwork\\Dataset"
    train_set = datasets.CIFAR100(root=DATA_ROOT, train=True, download=False, transform=TRAIN_TRANSFORM)
    test_set = datasets.CIFAR100(root=DATA_ROOT, train=False, download=False, transform=TEST_TRANSFORM)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )

    return train_loader, test_loader