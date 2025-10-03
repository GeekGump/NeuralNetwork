from Net.MyNet import MyNet  # 导入自己的网络
from Net.preresnet import preresnet110  # 导入预激活 ResNet18
from Net.resnet import resnet20
from Net.densenet import densenet100bc  # 导入 DenseNet121
from Net.MultiLayerGroupNet import MultiLayerGroupNet  # 导入多层分组网络
from enum import Enum
class NetType(Enum):
    MyNet = 1
    preresnet110 = 2
    densenet100bc = 3
    MultiLayerGroupNet = 4
    resnet20 = 5
    

def GetNet(NetType = NetType, num_classes=100):
    model_name = ""
    if NetType == NetType.MyNet:
        net = MyNet(num_classes=num_classes)
        model_name = "MyNet"
    elif NetType == NetType.preresnet110:
        net = preresnet110(num_classes=num_classes)
        model_name = "preresnet110"
    elif  NetType == NetType.densenet100bc:
        net = densenet100bc(num_classes=num_classes)
        model_name = "densenet100bc"
    elif NetType == NetType.resnet20:
        net = resnet20(num_classes=num_classes)
        model_name = "resnet20"
    elif NetType == NetType.MultiLayerGroupNet:
        net = MultiLayerGroupNet(n_layers=3, m_base=32, num_classes=num_classes, input_size=32)
        model_name = "MultiLayerGroupNet"
    else:
        raise ValueError("Unsupported network architecture: {}".format(NetType))
    return net, model_name

#         net = MultiLayerGroupNet(n_layers=3, m_base=32, num_classes=num_classes, input_size=32)