import torchvision
import torch.nn as nn

def resnet18(num):
    net = torchvision.models.resnet18()
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.conv1.weight.data = net.conv1.weight.data.sum(dim=1, keepdim=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num)
    return net

def vgg16(num):
    net = torchvision.models.vgg16()
    net.features[0] = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    num_ftrs = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(num_ftrs, num)
    return net

model_box = {
    'resnet' : resnet18,
    'vgg' : vgg16,
}

class MTypeError(Exception):
    """自定义异常类，用于处理 NaN 值的情况"""
    pass

def opt_backbone(mtype: str, task: str):
    try:
        if 'MSTAR_SOC' in task:
            func = model_box[mtype](10)
        elif 'MSTAR_EOC1' in task:
            func = model_box[mtype](4)
        elif 'MSTAR_EOC2' in task:
            func = model_box[mtype](4)
        elif 'SAR_ACD' in task:
            func = model_box[mtype](6)
        elif 'SRSDD' in task:
            func = model_box[mtype](5)
        else:
            raise MTypeError('there is no such task type')
    except KeyError:
        raise MTypeError('there is no such backbone type')
    return func