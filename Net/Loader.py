from enum import Enum

class NetType(Enum):
    MyNet = 1
    preresnet110 = 2
    densenet121 = 3
    MultiLayerGroupNet = 4
    resnet20 = 5
    senet = 6
    GroupConvClassifier = 7
    

def GetNet(NetType = NetType, num_classes=100):
    if NetType == NetType.MyNet:
        from Net.MyNet import MyNet  # 导入自己的网络
        net = MyNet(num_classes=num_classes)
    elif NetType == NetType.preresnet110:
        from Net.preresnet import preresnet110  # 导入预激活 ResNet18
        net = preresnet110(num_classes=num_classes)
    elif  NetType == NetType.densenet121:
        from Net.densenet import densenet121
        net = densenet121()
    elif NetType == NetType.resnet20:
        from Net.resnet import resnet20
        net = resnet20(num_classes=num_classes)
    elif NetType == NetType.MultiLayerGroupNet:
        from Net.MultiLayerGroupNet import MultiLayerGroupNet  # 导入多层分组网络
        net = MultiLayerGroupNet(n_layers=3, m_base=32, num_classes=num_classes, input_size=32)
    elif NetType == NetType.senet:
        from Net.senet import se_resnext29_8x64d
        net = se_resnext29_8x64d(num_classes=num_classes)
    elif NetType == NetType.GroupConvClassifier:
        from Net.GroupConvClassifier import GroupConvClassifier
        net = GroupConvClassifier(num_classes=num_classes)
    else:
        raise ValueError("Unsupported network architecture: {}".format(NetType))
    return net

#         net = MultiLayerGroupNet(n_layers=3, m_base=32, num_classes=num_classes, input_size=32)