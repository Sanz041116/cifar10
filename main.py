from datetime import datetime

import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


epoch_num = 150

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
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
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x



model = Model()
if torch.cuda.is_available():
    model = model.cuda()



# 损失函数
loss = nn.CrossEntropyLoss()
# 优化器 这需要改
optimizer  = torch.optim.SGD(model.parameters(),lr=0.01,)

i = 1 # 用于绘制测试集的tensorboard

#添加tensorboard可视化数据
time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer = SummaryWriter('./logs_tensorboard/'+time_stamp)

num_time = 0 # 记录看看每轮有多少次训练
for epoch in range(epoch_num):

    print('开始第{}轮训练'.format(epoch + 1))
    model.train()
    for data in train_dataloader:
        # 数据分开 一个是图片数据，一个是真实值
        imgs, targets = data
        imgs = imgs.cuda()  # 放到GPU上一会训练用
        targets = targets.cuda()
        # 拿到预测值
        output = model(imgs)
        # 计算损失值
        loss_in = loss(output, targets)
        # 优化开始~ ~ 先梯度清零
        optimizer.zero_grad()
        # 反向传播+更新
        loss_in.backward()
        optimizer.step()
        num_time += 1
        if num_time % 100 == 0:
            writer.add_scalar('训练集损失值', loss_in.item(), num_time)

    accurate = 0 #目前就准确率
    sum_loss = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            # 这里的每一次循环 都是一个minibatch  一次for循环里面有64(batch_size)个数据。
            imgs , targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = model(imgs)
            loss_in = loss(output,targets)

            sum_loss += loss_in
            output_temp = output.argmax(1)
            accurate += (output.argmax(1) == targets).sum()

    print('第{}轮测试集的正确率:{:.2f}%'.format(epoch + 1, accurate / len(test_data) * 100))

    writer.add_scalar('测试集损失', sum_loss, i)
    writer.add_scalar('当前测试集正确率', accurate / len(test_data) * 100, i)
    i += 1

    torch.save(model, './model_pytorch/model_{}.pth'.format(epoch + 1))

writer.close()
