import torch
from torchvision.models import vgg16  # 以 vgg16 为例
from Net.Loader import GetNet  # 导入自己的网络
from Net.Loader import NetType
from torch.autograd import Variable

model_name = "MultiLayerGroupNetV2"
model_path = "Model/" + model_name+ ".pth"
myNet = GetNet(NetType=NetType.MultiLayerGroupNet)  # 实例化 resnet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myNet.load_state_dict(torch.load(model_path,map_location=device))
myNet.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 32, 32)  # 随机生成一个输入
    modelData = "./Model/demo.pth"  # 定义模型数据保存的路径
    torch.onnx.export(myNet, x, modelData)


