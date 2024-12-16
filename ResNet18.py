from datetime import datetime

import torchvision
import torch
from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# 2024.12.3 新增数据增强-该情况下预测结果output极其的单一，总之结果不好
# myTransforms = transforms.Compose([
#     # transforms.Resize((224, 224)), # 尝试resize
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# 2024.12.5 新的数据增强方式
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

epoch_num = 130
batch_size = 64
learning_rate = 0.01
optimizer_select = 'SGD' #SGD,Momentum,Adadelta,Adam,Adamax,RMSprop
momentum = 0.9
beta1 = 0.9  # 一阶矩估计的指数衰减率
beta2 = 0.999  # 二阶矩估计的指数衰减率
alpha = 0.9

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train,
                                          download=False)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test,
                                         download=False)
# DataLoader加载数据集
'''
2024.12.5
    训练集增加shuffle
'''
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# print("训练集的长度:{}".format(len(train_data)))
# print("测试集的长度:{}".format(len(test_data)))

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize 对输入的图像数据进行反归一化处理。如果图像数据是归一化到[-1, 1]区间的，那么通过除以2然后加上0.5，可以将数据转换回[0, 1]区间，这样就可以正确地显示图像的颜色。
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0))) # 转换维度顺序为matplot所需顺序（H，W，C）
#     plt.show()


# 随机选取图片，因为trainloader是随机的
# dataiter = iter(train_dataloader)
# images, labels = dataiter.next()

# 显示图片
# imshow(torchvision.utils.make_grid(images))  # make_grid的作用是将若干幅图像拼成一幅图像。

# 搭建神经网络
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # 这个1x1卷积层用于匹配输入和输出的维度，以便可以将输入直接添加到主体部分的输出上
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 例如：strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    model = ResNet(ResidualBlock)
    if torch.cuda.is_available():
        model = model.to(device)

    # 损失函数
    loss = nn.CrossEntropyLoss()
    # 优化器 这需要改
    optimizer  = torch.optim.SGD(model.parameters(),lr=learning_rate,)
    # optimizer  = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate,betas=(beta1, beta2))
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate,alpha=alpha)

    i = 1  # 用于绘制测试集的tensorboard

    # 添加tensorboard可视化数据
    time_stamp = "{0:%Y-%m-%dT%H-%M/}".format(datetime.now())
    # writer = SummaryWriter('./logs_tensorboard/'+time_stamp,optimizer_select+'_lr='+str(learning_rate)+'_epoch='+str(epoch_num)+'/')
    writer = SummaryWriter(f'./logs_tensorboard/{batch_size}/{optimizer_select}_lr={learning_rate}_epoch={epoch_num}/')
    # writer = SummaryWriter(f'./logs_tensorboard/{batch_size}/{optimizer_select}_lr={learning_rate}_epoch={epoch_num}_alpha={alpha}/')
    # writer = SummaryWriter(
    #     f'./logs_tensorboard/{batch_size}/{optimizer_select}_lr={learning_rate}_epoch={epoch_num}_betas={beta1,beta2}/')
    num_time = 0  # 记录看看每轮有多少次训练
    for epoch in range(epoch_num):
        sum_loss = 0
        print('开始第{}轮训练'.format(epoch + 1))
        model.train()
        for data in train_dataloader:
            # 数据分开 一个是图片数据，一个是真实值
            imgs, targets = data
            imgs = imgs.to(device)  # 放到GPU上一会训练用
            targets = targets.to(device)
            # 拿到预测值
            output = model(imgs)
            # 计算损失值
            loss_in = loss(output, targets)
            sum_loss += loss_in.item()
            # 优化开始~ ~ 先梯度清零
            optimizer.zero_grad()
            # 反向传播+更新
            loss_in.backward()
            optimizer.step()
            num_time += 1
            if num_time % 200 == 0:
                writer.add_scalar('训练集损失值', loss_in.item(), num_time)

        batch_num = len(train_dataloader)
        writer.add_scalar('训练集损失值(per epoch)', sum_loss/batch_num, i)

        # 五个评估指标
        accuracy = 0
        precision = 0
        recall = 0
        f1score = 0
        kappa = 0

        sum_loss = 0

        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                # 这里的每一次循环 都是一个minibatch  一次for循环里面有64(batch_size)个数据。
                imgs, target = data
                imgs, target = imgs.to(device), target.to(device)
                output = model(imgs)  # model在gpu上
                loss_in = loss(output, target)

                output = output.argmax(1).cpu().numpy()
                target = target.cpu().numpy()

                '''
                classification_report 会得到每一个类的指标
                如下： 
               precision    recall  f1-score   support

           0       1.00      0.50      0.67         4
           1       0.50      0.67      0.57         3
           2       0.33      0.50      0.40         2
        。。。
    accuracy                           0.56         9
   macro avg       0.61      0.56      0.55         9
weighted avg       0.69      0.56      0.58         9

                '''
                # accuracy += classification_report(target, output, output_dict=True, zero_division=0)['accuracy']
                # s = classification_report(target, output, output_dict=True, zero_division=0)['weighted avg']
                # precision += s['precision']
                # recall += s['recall']
                # f1_score += s['f1-score']
                # kappa += cohen_kappa_score(target, output)

                '''
                sklearn计算评估指标的代码，上面的代码忘记从哪来了，反正都试试
                问：现在的实验结果中指标曲线极为相似（特别是准确率和召回率完全相同）为什么？
                答：可能是因为数据集中的正负样本数量相等，或者模型在各个类别上的预测结果分布非常均匀，那么准确率（整体预测正确的比例）和召回率（正样本被正确识别的比例）可能会相同。这是因为在这种情况下，模型对每个类别的预测表现是一致的
                    or https://blog.csdn.net/fujikoo/article/details/119926390的解释
                    or 要不试试看把混淆矩阵弄出来
                '''
                accuracy += accuracy_score(target, output,)
                precision += precision_score(target,output,average='macro',zero_division=0)
                recall += recall_score(target,output,average='macro',zero_division=0)
                f1score += f1_score(target,output,average='macro',zero_division=0)
                kappa += cohen_kappa_score(target,output)

                sum_loss += loss_in.item()

                # accurate += (output.argmax(1) == targets).sum()

        # print('第{}轮测试集的正确率:{:.2f}%'.format(epoch + 1, accurate / len(test_data) * 100))
        # batch_num = len(test_data) // batch_size + 1
        batch_num = len(test_dataloader)
        writer.add_scalar('测试集损失', sum_loss, i)
        writer.add_scalar('测试集损失(平均损失)', sum_loss/batch_num, i)  # 这个我分析不来，不要也行

        # 2024.12.3 改用sklearn库计算评估指标

        writer.add_scalar('测试集准确率', accuracy / batch_num, i)
        writer.add_scalar('测试集精确率', precision / batch_num, i)
        writer.add_scalar('测试集召回率', recall / batch_num, i)
        writer.add_scalar('测试集F1值', f1score / batch_num, i)
        writer.add_scalar('测试集kappa指标', kappa / batch_num, i)
        # writer.add_scalar('当前测试集准确率', accurate / len(test_data) * 100, i)

        i += 1

        # if epoch % 10 == 0:
        #     torch.save(model, f'./model_pytorch/Resnet/{optimizer_select}_lr={learning_rate}_epoch={epoch_num}_{epoch+1}.pth')
    torch.save(model.state_dict(), f'./model_pytorch/Resnet/{batch_size}/{optimizer_select}_lr={learning_rate}_epoch={epoch_num}.pth')
    # torch.save(model.state_dict(),
    #            f'./model_pytorch/Resnet/{batch_size}/{optimizer_select}_lr={learning_rate}_epoch={epoch_num}_alpha={alpha}.pth')
    # torch.save(model.state_dict(),
    #            f'./model_pytorch/Resnet/{batch_size}/{optimizer_select}_lr={learning_rate}_epoch={epoch_num}_betas={beta1,beta2}.pth')
    writer.close()
