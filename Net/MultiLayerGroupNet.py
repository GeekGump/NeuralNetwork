import torch
import torch.nn as nn

class FunctionalModule(nn.Module):
    def __init__(self, in_channels):
        super(FunctionalModule, self).__init__()
        # 双重卷积 + ReLU（保持通道和尺寸不变）
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)  # 双重卷积后加ReLU，增强非线性
        )
        # 添加SE注意力机制
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        indentity = x  # 残差连接
        out = self.double_conv(x)
        # SE注意力权重
        se_weight = self.se(out)
        out = out * se_weight

       
        return self.relu(out + indentity)  # 残差连接
    
class MultiLayerGroupNet(nn.Module):
    def __init__(self, n_layers, m_base, num_classes, input_size=32):
        """
        多层分组网络
        
        参数说明：
            n_layers (int): 层级数（分组处理的层数，如n=3则有3层）
            m_base (int): 每组的基础通道数（如m=32，则第一层每组32通道）
            num_classes (int): 分类类别数（如10类）
            input_size (int): 输入图像的空间尺寸（假设为正方形，如32x32）
        """
        super(MultiLayerGroupNet, self).__init__()
        self.n_layers = n_layers
        self.m_base = m_base
        self.input_size = input_size
        
        # --------------------------
        # 1. 初始卷积：3通道 → 2^(n-1)*m_base通道（保持尺寸不变）
        # --------------------------
        initial_channels = 2 ** (n_layers - 1) * m_base
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, initial_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_channels//2, initial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_channels),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(initial_channels, initial_channels, kernel_size=1, padding=0)
        )
        self.initial_relu = nn.ReLU(inplace=True)  # 激活函数
        
        # --------------------------
        # 2. 分层分组处理：n层，每层模块数从2^(n-1)降到1
        # --------------------------
        self.layer_modules = nn.ModuleList()  # 存储所有层的分组模块
        for layer_idx in range(n_layers):
            # 当前层的模块数：Gl = 2^(n - (layer_idx+1))（layer_idx从0开始）
            gl = 2 ** (n_layers - (layer_idx + 1))
            # 当前层每组的通道数：cl = 总通道 / 组数（保证总通道不变）
            total_channels = initial_channels
            cl = total_channels // gl
            
            # 当前层的模块列表：每个模块处理cl通道
            layer_module = nn.ModuleList([
                FunctionalModule(in_channels=cl) for _ in range(gl)
            ])
            self.layer_modules.append(layer_module)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        # --------------------------
        # 3. 分类器：展平特征图 → Linear输出分类数
        # --------------------------
        # 最后一层在经过 self.avgpool((2,2)) 后的特征图尺寸：(batch, initial_channels, 2, 2)
        final_feat_size = initial_channels * 2 * 2
        self.classifier = nn.Sequential(
            nn.Linear(final_feat_size, final_feat_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_feat_size//2, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # --------------------------
        # 步骤1：初始卷积
        # --------------------------
        x = self.initial_conv(x)  # (batch, 2^(n-1)*m, H, W)
        x = self.initial_relu(x)

        # --------------------------
        # 步骤2：逐层分组处理
        # --------------------------
        for layer_idx in range(self.n_layers):
            # 获取当前层的模块列表、组数、每组通道数
            current_layer = self.layer_modules[layer_idx]
            gl = len(current_layer)  # 当前层模块数（如n=3时，第一层gl=4）
            total_channels = x.size(1)  # 当前总通道数（始终不变）
            cl = total_channels // gl  # 当前每组通道数

            # 按通道维度分组（将total_channels分成gl组，每组cl通道）
            x_groups = torch.chunk(x, gl, dim=1)  # 列表，每个元素是(batch, cl, H, W)

            # 每组通过对应的FunctionalModule
            processed_groups = []
            for group_idx in range(gl):
                out = current_layer[group_idx](x_groups[group_idx])  # 输出(batch, cl, H, W)
                processed_groups.append(out)

            # 拼接处理后的组，恢复总通道数（gl * cl = total_channels）
            x = torch.cat(processed_groups, dim=1)  # (batch, total_channels, H, W)

        # --------------------------
        # 步骤3：分类输出（先自适应平均池化到固定尺寸，再展平并分类）
        # --------------------------
        x = self.avgpool(x)  # 将空间尺寸池化到 (2,2)
        x = x.view(batch_size, -1)  # 展平特征图：(batch, total_channels*2*2)
        logits = self.classifier(x)  # (batch, num_classes)

        return logits