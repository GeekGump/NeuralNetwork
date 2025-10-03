import torch
import torch.nn as nn
import torch.nn.functional as F

class FunctionalModule(nn.Module):
    def __init__(self):
        super(FunctionalModule, self).__init__()
        self.func = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=3, padding=0),
             nn.BatchNorm2d(8),
            nn.Flatten(),
            nn.Linear(32*6*6, 9)
        )

    def forward(self, x):
        return self.func(x)
    

class MyNet(nn.Module):
    def __init__(self, num_classes=100):
        super(MyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.func = nn.ModuleList([FunctionalModule() for _ in range(64)])

        self.classifier = nn.Sequential(
            nn.Dropout(),
            #nn.Linear(128*6*6, 512),
            nn.Linear(576 , 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        feat_chunks = torch.chunk(feat, chunks=64, dim=1)

        processed_chunks = []
        for idx, module in enumerate(self.func):
            # 取第idx组特征，输入第idx个FunctionalModule
            out = module(feat_chunks[idx])  # 每个out形状：(batch_size, 64, H''=H'/2, W''=W'/2)
            processed_chunks.append(out)
        x = torch.cat(processed_chunks, dim=1)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x