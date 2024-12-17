import torch
import torchvision
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
from ResNet18 import ResNet, ResidualBlock, batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

epoch_num = 130
batch_size = 64
learning_rate = 0.001
optimizer_select = 'RMSprop' #SGD,Momentum,Adadelta,Adam,Adamax,RMSprop
momentum = 0.9
beta1 = 0.9  # 一阶矩估计的指数衰减率
beta2 = 0.999  # 二阶矩估计的指数衰减率
alpha = 0.9

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test,
                                         download=False)
test_dataloader = DataLoader(test_data,batch_size=len(test_data))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



if __name__ == '__main__':
    # model = torch.load(f'./model_pytorch/Resnet/{optimizer_select}_lr={learning_rate}_epoch={epoch_num}_betas={beta1,beta2}.pth')

    model = ResNet(ResidualBlock)
    model.load_state_dict(torch.load(
        f'./model_pytorch/Resnet/{optimizer_select}_lr={learning_rate}_epoch={epoch_num}_alpha={alpha}.pth'))
    # model.load_state_dict(torch.load(f'./model_pytorch/Resnet/{optimizer_select}_lr={learning_rate}_epoch={epoch_num}_betas=({beta1},{beta2}).pth'))
    model.to(device)
    model.eval()
    with torch.no_grad():
            # 这里的每一次循环 都是一个minibatch  一次for循环里面有64(batch_size)个数据。
        for data in test_dataloader:
            imgs, target = data
            imgs, target = imgs.to(device), target.to(device)
            output = model(imgs)  # model在gpu上

            output = output.argmax(1).cpu().numpy()
            target = target.cpu().numpy()
            report = classification_report(target, output, zero_division=0, target_names=classes,labels=range(len(classes)),digits=4)
            print(report)