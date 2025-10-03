from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import Net.Loader as Loader
from Net.Loader import NetType
import torch
import time
import matplotlib.pyplot as plt
import torchvision
import os
import datetime

model_name = "MultiLayerGroupNetV2"
model = Loader.GetNet(NetType=NetType.MultiLayerGroupNet, num_classes=100)
model_path = "Model/" + model_name+ ".pth"

import torch_directml
device = torch_directml.device()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

# 测试集预处理（无需数据增强）
test_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

print("Loading data...")
# data_root = "CIFAR-100-dataset-main"
# train_data = CIFAR100CustomDataset(root=data_root, split="train", transforms=train_transforms)
# test_data = CIFAR100CustomDataset(root=data_root, split="test", transforms=test_transforms)

 
test_dataset = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,                              # 加载测试集
    download=False,
    transform=test_transforms
)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Correct/Total:  "+str(correct)+"/" + str(total))
accuracy = 100 * correct / total
print(accuracy)