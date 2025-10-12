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
import time

start_time = time.time()
model,model_name = Loader.GetNet(NetType=NetType.densenet121, num_classes=100)
model_path = "Model/" + model_name+ ".pth"

# set up device and data root
device = torch.device("cpu")
data_root = "./data"
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    import torch_directml
    device = torch_directml.device()
    data_root = "D:\\SHIVAAAA\\Documents\\MinGW\\NeuralNetwork\\Dataset"
print(device)

# if model_path exists, load the model
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
else:
    print("Model not found!")
    exit()
model = model.to(device)
model.eval()

# 测试集预处理（无需数据增强）
test_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

print("Loading data...")
 
test_dataset = torchvision.datasets.CIFAR100(
    root=data_root,
    train=False,                              # 加载测试集
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
end_time = time.time()
print("Time taken: "+str(end_time-start_time)+" seconds")
print("Correct/Total:  "+str(correct)+"/" + str(total))
accuracy = 100 * correct / total
print("Accuracy: "+str(accuracy))