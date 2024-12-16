import netron
import onnxoptimizer
import torch
import torchvision

from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import transforms
import onnx
from torchviz import make_dot

from ResNet18 import ResNet,ResidualBlock

batch_size=64

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train,
                                          download=True)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# model = ResNet(ResidualBlock)
if __name__ == '__main__':
    # input,target = next(iter(train_dataloader))
    # print(input.shape)
    # input = torch.ones((64, 3, 32, 32))
    model = ResNet(ResidualBlock)
    # torch.onnx.export(model, input, f='ResNet18.onnx')

    # 加载修剪之前的模型
    # onnx_model = onnx.load(r"ResNet18.onnx")

    # 去掉identity层
    # all_passes = onnxoptimizer.get_available_passes()
    # print("Available optimization passes:")
    # for p in all_passes:
    #     print('\t{}'.format(p))
    # print()

    # 保存修剪后的模型
    # onnx_optimized = 'Resnet18-simplify.onnx'
    # passes = ['eliminate_identity']
    # optimized_model = onnxoptimizer.optimize(onnx_model, passes)
    # onnx.save(optimized_model, onnx_optimized)

    # 使用torchsummary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary(model, (3,32,32),batch_size=batch_size)

    # 使用torchviz 没用懂
    # output = model(input)
    # g = make_dot(output)
    # g.view()

    netron.start('Resnet18-simplify.onnx')