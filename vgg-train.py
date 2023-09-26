import torch
import torch.nn as nn
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64, 64, 3, 1, 1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),nn.BatchNorm2d(128),nn.ReLU(),nn.Conv2d(128, 128, 3, 1, 1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1),nn.BatchNorm2d(256),nn.ReLU(),nn.Conv2d(256, 256, 3, 1, 1),nn.BatchNorm2d(256),nn.ReLU(),nn.Conv2d(256, 256, 3, 1, 1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1),nn.BatchNorm2d(512),nn.ReLU(),nn.Conv2d(512, 512, 3, 1, 1),nn.BatchNorm2d(512),nn.ReLU(),nn.Conv2d(512, 512, 3, 1, 1),nn.BatchNorm2d(512),nn.ReLU(),nn.MaxPool2d(2, 2))
        self.layer5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),nn.BatchNorm2d(512),nn.ReLU(),nn.Conv2d(512, 512, 3, 1, 1),nn.BatchNorm2d(512),nn.ReLU(),nn.Conv2d(512, 512, 3, 1, 1),nn.BatchNorm2d(512),nn.ReLU(),nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Flatten(),nn.Linear(512, 512),nn.ReLU(),nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(512, 256),nn.ReLU(),nn.Dropout())
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

import torchvision
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
#from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = torchvision.datasets.CIFAR10(root="data", train=True, transform=torchvision.transforms.ToTensor(),download=False)
test_data = torchvision.datasets.CIFAR10(root="data", train=False, transform=torchvision.transforms.ToTensor(),download=False)
train_dataloader = DataLoader(train_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))
# 创建网络模型
vgg16 = VGG16()
vgg16 = vgg16.to(device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器
learning_rate = 0.015  # 设置学习速率
optimizer = torch.optim.SGD(vgg16.parameters(), lr=learning_rate)
# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 40
for i in range(epoch):
    print("--------第{}轮训练开始---------".format(i + 1))
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = vgg16(imgs)
        loss = loss_fn(outputs, targets)
        # 梯度调0
        optimizer.zero_grad()
        # 反向传播 梯度
        loss.backward()
        # 调优
        optimizer.step()
        #记录训练次数
        total_train_step = total_train_step + 1
        # 每100打印loss
        if total_train_step % 100 ==0:print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
    # 测试，没梯度没有调优代码
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs=imgs.cuda()
                targets=targets.cuda()
            outputs = vgg16(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            # 计算整体测试集上的正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

        print("整体测试集上的loss：".format(total_test_loss))
        # 使用/进行的tensor整数除法不再支持，可以使用true_divide代替
        print("整体测试集上的正确率：{}".format(total_accuracy.true_divide(test_data_size)))
        total_test_step = total_test_step + 1
        # 可视化正确率
# 保存每一轮的模型 这是第一种保存方式非官方推荐
torch.save(vgg16, "vgg16.pt")
print("模型已保存")

