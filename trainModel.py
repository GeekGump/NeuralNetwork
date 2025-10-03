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

model,model_name = Loader.GetNet(NetType=NetType.MultiLayerGroupNet, num_classes=100)
model_path = "Model/" + model_name+ ".pth"
load_model = False

# set up device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    import torch_directml
    device = torch_directml.device()
print(device)

# if model_path exists, load the model
if os.path.exists(model_path) and load_model:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = model.to(device)
model.train()

# 数据增强
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
])

# 测试集预处理（无需数据增强）
test_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

print("Loading data...")
train_dataset = torchvision.datasets.CIFAR100(
    root='./data',                            # 数据集下载路径
    train=True,                               # 加载训练集
    download=False,                            # 如果本地没有则下载
    transform=train_transforms                       # 应用预处理
)
 
test_dataset = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,                              # 加载测试集
    download=False,
    transform=test_transforms
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
# torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
# StepLR(optimizer, step_size=10, gamma=0.1)
num_epochs = 30
best_accuracy = 0

loss_history = []
accuracy_history = []


for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0

    start_time = time.time()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    scheduler.step()
    end_time = time.time()
    print(f"Training time for epoch {epoch+1}: {end_time - start_time:.2f} seconds")
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    loss_history.append(epoch_loss)
    
    # 每个 epoch 后进行一次测试集评估
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
    
    accuracy = 100 * correct / total
    accuracy_history.append(accuracy)
    print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%")


    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f"New best accuracy: {best_accuracy:.2f}%, saving model...")
        torch.save(model.state_dict(), model_path)
        
    formatted_loss = [f"{x:.2f}" for x in loss_history]
    formatted_loss = [float(x) for x in formatted_loss]
    print("loss_history:",formatted_loss[-10:])  # 输出：['0.12', '0.65', '0.99', '0.46']
    print("accuracy_history (last 10):", accuracy_history[-10:])
    print("  ")


def plot_float_array( 
    float_array, 
    title="Float Array Trend", 
    save_path=None,
    figsize=(8, 4),
    marker='o',
    linestyle='-',
    color='b',
    grid=True
 ):
    plt.figure(figsize=figsize)
    plt.plot(float_array, marker=marker, linestyle=linestyle, color=color, label='Value')
    plt.xlabel('Index')
    plt.ylabel('Float Value')
    plt.title(title)
    if grid:
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 自动创建目录
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")
    plt.close()


def save_float_lists_to_txt(name,list1, list2, output_path="float_data.txt"):
    # 检查两个列表长度是否一致
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度不一致！")
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

    # 定义格式化函数：保留两位有效数字
    def format_float(x):
        return f"{float(f'{x:.2g}')}"  # 先用'.2g'保留两位有效数字，再转float避免科学计数法
    
    # 写入文件
    with open(output_path, 'a', encoding='utf-8') as f:
        # 写入时间戳
        f.write(f"\n")
        f.write(f"# {name}\n")
        f.write(f"# {timestamp}\n")
        
        # 写入第一个列表（一行）
        f.write("loss rate  ")
        f.write(" ".join(format_float(x) for x in list1) + "\n")
        
        # 写入第二个列表（一行）
        f.write("accuracy   ")
        f.write(" ".join(format_float(x) for x in list2) + "\n")
    
    print(f"数据已保存至: {output_path}\n")


# 调用函数保存数据
save_float_lists_to_txt(
    "Example Data",
    list1=loss_history,
    list2=accuracy_history,
    output_path="./Record/record.txt"  # 自定义路径
)

plot_float_array(
    loss_history, 
    title=model_name+"Training Loss Over Epochs", 
    save_path="Record/"+ model_name +"loss_curve.png",
    color='blue'
)
plot_float_array(
    accuracy_history, 
    title=model_name+"Test Accuracy Over Epochs", 
    save_path="Record/"+ model_name +"accuracy_curve.png",
    color='red'
)