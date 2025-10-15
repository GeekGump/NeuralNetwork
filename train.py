import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import datetime
from tqdm import tqdm
from utils import get_dataloaders  # 用户自定义的获取DataLoader函数
import Net.Loader as Loader
from Net.Loader import NetType
import os
import json
import matplotlib.pyplot as plt

# ---------------------- 核心可配置区（需修改的参数集中在此处） ----------------------
# 数据集配置
IS_LOAD_HISTORY = False           # 是否加载历史记录（如需从头开始训练改为False）   
  
NET_TYPE = NetType.GroupConvClassifier  # 网络类型（对应Net.Loader中的枚举）
MODEL_NAME = NET_TYPE.name  # 模型名称（用于保存文件命名）
NUM_CLASSES = 100

# 训练超参数
BATCH_SIZE = 128                # 根据GPU内存调整（如GPU显存小可改为64）
EPOCHS = 40                     # 总训练轮次
LEARNING_RATE = 1e-1            # 初始学习率
WEIGHT_DECAY = 5e-4             # L2正则化系数
MILE_STONES = [25, 30,35]          # 学习率调整里程碑
GAMMA = 0.2                    # 学习率调整衰减系数
# 模型与设备
MODEL_GETTER =  Loader.GetNet(NetType=NET_TYPE, num_classes=100)

# ---------------------- 其他配置（一般不需修改） ----------------------
net = MODEL_GETTER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones= MILE_STONES,gamma=GAMMA)


CHECKPOINT_FILE = f"./CheckPoint/checkpoint_{MODEL_NAME}.pth"  # Checkpoint文件路径
os.makedirs("./CheckPoint", exist_ok=True)  # 确保Checkpoint文件夹存在
MODEL_PATH = f"./Model/{MODEL_NAME}.pth"  # 最佳模型文件路径
os.makedirs("./Model", exist_ok=True)  # 确保模型文件夹存在
RECORD_FOLDER = "./Record"          # 训练记录文件夹
os.makedirs(RECORD_FOLDER, exist_ok=True)  # 确保记录文件夹存在
HISTORY_FILE = f"{RECORD_FOLDER}/history.json"      # 训练历史文件路径

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    import torch_directml
    device = torch_directml.device()
DEVICE = device
print(device)

def train_model(
    net: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    epochs: int,
    device: torch.device,
    is_load_history: bool = IS_LOAD_HISTORY
):
    """
    模型训练主循环（含进度条、时间估计、指标打印）
    :param net: 待训练模型
    :param train_loader: 训练数据加载器
    :param test_loader: 测试数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param epochs: 总轮次
    :param device: 训练设备
    """
    net.to(device)
    best_val_acc = 0.0  # 记录最佳验证准确率
    start_epoch = 0    # 记录起始轮次（断点续训时更新）
    best_val_acc_epoch = 0  # 记录最佳准确率对应的轮次
    loss_history = []  # 记录每轮训练损失
    acc_history = []   # 记录每轮验证准确率
    # -------------------- 加载历史与断点 --------------------
    current_net_history = {}
    if is_load_history:
        print("Loading training history...")

        history = load_history()
        if MODEL_NAME not in history:
            print(f"No history found for model {MODEL_NAME}. Starting fresh.")
            history[MODEL_NAME] = {
                "trained_epoch": 0,
                "train_losses": [],
                "val_accs": [],
                "best_val_acc": 0.0,
                "best_epoch": 0
            }
            save_history(history)
        current_net_history = history.get(MODEL_NAME, {
            "trained_epoch": 0,          # 训练过的轮次（从1开始）
            "train_losses": [],    # 每轮训练平均Loss
            "val_accs": [],        # 每轮验证平均准确率
            "best_val_acc": 0.0,   # 最佳验证准确率
            "best_epoch": 0        # 最佳准确率对应的轮次
        })
        # 加载断点（模型、优化器、epoch、best_val_acc）
        start_epoch, best_val_acc = load_checkpoint(net, optimizer)
        loss_history = current_net_history["train_losses"]
        acc_history = current_net_history["val_accs"]
        best_val_acc = current_net_history.get("best_val_acc", 0.0)
        best_val_acc_epoch = current_net_history.get("best_epoch", 0)
        print(f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.2f}% at epoch {best_val_acc_epoch}")
    else:
        history = load_history()
        history[MODEL_NAME] = {
            "trained_epoch": 0,
            "train_losses": [],
            "val_accs": [],
            "best_val_acc": 0.0,
            "best_epoch": 0
        }
        current_net_history = history[MODEL_NAME]
        loss_history = current_net_history["train_losses"]
        acc_history = current_net_history["val_accs"]
        save_history(history)
        

    for epoch in range(start_epoch,epochs):
        train_start_time = time.time()  # 记录本轮开始时间
        # -------------------- 训练阶段 --------------------
        avg_train_loss, avg_train_acc = train_one_epoch(
            net, train_loader, criterion, optimizer, scheduler, epoch, epochs, device
        )
        # -------------------- 验证阶段 -------------------- 
        avg_val_loss, avg_val_acc = evaluate_model(
            net, test_loader, criterion, device
        )

        # -------------------- 时间统计与剩余时间估计 -------------------- 
        epoch_elapsed = time.time() - train_start_time  # 本轮总耗时（秒）
        remaining_epochs = epochs - epoch - 1  # 剩余轮次
        remaining_seconds = remaining_epochs * epoch_elapsed  # 剩余总秒数
        
        # 转换为可读时间格式（时:分:秒）
        remaining_time = datetime.timedelta(seconds=int(remaining_seconds))
        fake_time = (datetime.datetime.min + remaining_time).time()
        formatted_time = fake_time.strftime("%H:%M")
        hours,minutes = formatted_time.split(":")
        if hours == "00":
            formatted_time = f"{int(minutes)} min"
        else:
            formatted_time = f"{int(hours)} hr {int(minutes)} min"
        # 预计完成时间（当前时间 + 剩余时间）
        finish_time = datetime.datetime.now() + datetime.timedelta(seconds=remaining_seconds)
        finish_time_str = finish_time.strftime("%H:%M")

        # -------------------- 打印本轮总结 -------------------- 
        print(f"{'='*50}")
        print(f"Epoch [{epoch+1}/{epochs}] Summary:")
        print(f"    Train | Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2f}%")
        print(f"    Val   | Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.2f}%")
        print(f"    Time  | Epoch: {epoch_elapsed:.2f}s | Remaining: {formatted_time} | Finish at: {finish_time_str}")


        # -------------------- 更新历史记录 --------------------
        # loss_history.append(avg_train_loss)
        # acc_history.append(avg_val_acc)

        current_net_history["trained_epoch"] = epoch + 1
        current_net_history["train_losses"].append(avg_train_loss)
        current_net_history["val_accs"].append(avg_val_acc)
        print(f"train loss(last 10): {[round(l, 2) for l in loss_history[-10:]]}")
        print(f"val   acc(last 10): {acc_history[-10:]}")
        print(f"{'='*50}")

        # 更新最佳准确率与轮次
        if avg_val_acc > current_net_history["best_val_acc"]:
            current_net_history["best_val_acc"] = avg_val_acc
            current_net_history["best_epoch"] = epoch+1

        # 保存历史到JSON
        history[MODEL_NAME] = current_net_history
        save_history(history)
        save_checkpoint(net, optimizer, epoch, best_val_acc)

        # -------------------- 保存最佳模型 -------------------- 
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(
                net.state_dict(), 
                MODEL_PATH
            )
            print(f"    [Save] Best model updated with Val Acc: {best_val_acc:.2f}%")
        print()  # 换行
    plot_float_array(
        loss_history, 
        title=MODEL_NAME+"  Training Loss Over Epochs", 
        save_path=f"{RECORD_FOLDER}/{MODEL_NAME}_loss_curve.png",
        color='blue'
    )
    plot_float_array(
        acc_history, 
        title=MODEL_NAME+"  Test Accuracy Over Epochs", 
        save_path=f"{RECORD_FOLDER}/{MODEL_NAME}_accuracy_curve.png",
        color='red'
    )
    # 训练结束总结
    print("\n" + "="*50)
    print(f"Training Finished!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved at: {MODEL_PATH}")
    print("="*50)


def train_one_epoch(
    net: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    epoch: int,
    epochs: int,
    device: torch.device
):
  # -------------------- 训练阶段 -------------------- 
        net.train()
        total_train_loss = 0.0
        total_train_correct = 0
        train_start_time = time.time()  # 记录本轮开始时间

        # 带进度条的训练循环
        train_bar = tqdm(
            train_loader, 
            desc=f"[Epoch {epoch+1}/{epochs}] Training", 
            leave=False,  # 训练进度条完成后不保留
            dynamic_ncols=True  # 自动调整进度条宽度
        )
        for batch_idx, (imgs, labels) in enumerate(train_bar):
            imgs, labels = imgs.to(device), labels.to(device)

            # 前向传播
            logits = net(imgs)
            loss = criterion(logits, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练指标
            total_train_loss += loss.item()
            pred_labels = torch.argmax(logits, dim=1)
            total_train_correct += (pred_labels == labels).sum().item()


        scheduler.step()  # 更新学习率
        # 计算训练集平均指标
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_correct / len(train_loader.dataset) * 100
        return avg_train_loss, avg_train_acc


def evaluate_model(
    net: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """
    模型评估函数（在测试集上评估）
    :param net: 待评估模型
    :param test_loader: 测试数据加载器
    :param criterion: 损失函数
    :param device: 训练设备
    """
    net.to(device)
    net.eval()
    total_test_loss = 0.0
    total_test_correct = 0

    with torch.no_grad():  # 关闭梯度计算
        test_bar = tqdm(
            test_loader, 
            desc="Evaluating", 
            leave=False,
            dynamic_ncols=True
        )
        for imgs, labels in test_bar:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = net(imgs)
            loss = criterion(logits, labels)

            total_test_loss += loss.item()
            pred_labels = torch.argmax(logits, dim=1)
            total_test_correct += (pred_labels == labels).sum().item()

    # 计算测试集平均指标
    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_acc = total_test_correct / len(test_loader.dataset) * 100

    return avg_test_loss, avg_test_acc


def save_history(history: dict, file_path: str = HISTORY_FILE):
    """保存训练历史到JSON文件（覆盖写入）"""
    with open(file_path, "w") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


def load_history(file_path: str = HISTORY_FILE) -> dict:
    """从JSON文件加载训练历史（文件不存在时返回空字典）"""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
        return json.load(f)
    

def save_checkpoint(
    net: nn.Module, 
    optimizer: optim.Optimizer, 
    epoch: int, 
    best_val_acc: float, 
    file_path: str = CHECKPOINT_FILE
):
    """保存模型、优化器、epoch、最佳准确率到Checkpoint文件"""
    torch.save({
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc
    }, file_path)


def load_checkpoint(
    net: nn.Module, 
    optimizer: optim.Optimizer, 
    file_path: str = CHECKPOINT_FILE
) -> tuple[int, float]:
    """从Checkpoint文件加载模型、优化器、epoch、最佳准确率（文件不存在时返回初始状态）"""
    if not os.path.exists(file_path):
        print("No checkpoint found. Starting training from scratch.")
        return 0, 0.0
    checkpoint = torch.load(file_path, map_location=DEVICE)
    net.load_state_dict(checkpoint["net_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1  # 从下一轮开始
    best_val_acc = checkpoint["best_val_acc"]
    print(f"Loaded checkpoint from epoch {epoch-1}. Resuming training...")
    return epoch, best_val_acc


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


if __name__ == "__main__":
    # -------------------- 初始化资源 -------------------- 
    # 获取数据加载器
    print("Loading Data...")
    train_loader, test_loader = get_dataloaders(BATCH_SIZE,DEVICE)
    print(f"Data ready")
    # -------------------- 启动训练 -------------------- 
    train_model(
        net=net,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        device=DEVICE
    )