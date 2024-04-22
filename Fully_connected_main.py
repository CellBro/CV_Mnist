# 训练+测试


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都不同

################################################################################
# 超参数
EPOCH = 1  # 训练整批数据的次数
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False
#################################################################################

# 下载mnist手写数据集
train_data = torchvision.datasets.MNIST(
    root='./data/',  # 保存或提取的位置  会放在当前文件夹中
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray

    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False  # 表明是测试集
)

# 批训练 50个samples， 1  channel，28x28 (50,1,28,28)
# Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  # 是否打乱数据，一般都打乱
)

# 进行测试
# 为节约时间，测试时只测试前2000个

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
# torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
test_y = test_data.targets[:2000]


# 用class类来建立FullyConnectedNet模型
# CNN流程：卷积(Linear)-> 激励函数(ReLU)->
#         卷积(Linear)-> 激励函数(ReLU)->
#         卷积(Linear)-> 输出
#        展平多维的卷积成的特征图->接入全连接层(Linear)->输出

class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        # 第一个全连接层接受输入 784 个特征 (对应于 1x28x28 图像)
        self.fc1 = nn.Linear(784, 128)  # 从784个输入到128个输出
        self.fc2 = nn.Linear(128, 64)   # 第二个全连接层，从128个节点到64个节点
        self.fc3 = nn.Linear(64, 10)    # 最后一个全连接层，输出10个类别（对应于10个数字）

    def forward(self, x):
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # 展平图像,view就是把x弄成batchsize行个tensor，-1参数自动计算
        x = self.fc1(x)  # 第一层全连接
        x = nn.ReLU()(x)  # 激活函数
        x = self.fc2(x)  # 第二层全连接
        x = nn.ReLU()(x)  # 激活函数
        x = self.fc3(x)  # 最后一层全连接
        return x

# 选择网络
cnn = FullyConnectedNet()
# print(cnn)

# 训练
# 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差

# 优化器选择Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

# 开始训练
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
        output = cnn(b_x)  # 先将数据放到cnn中计算output
        loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
        optimizer.zero_grad()  # 清除之前学到的梯度的参数
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 应用梯度

        if step % 50 == 0: #每一个batch输出一次结果
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

torch.save(cnn.state_dict(), 'cnn2.pkl')#保存模型

# 加载模型，调用时需将前面训练及保存模型的代码注释掉，否则会再训练一遍
cnn.load_state_dict(torch.load('cnn2.pkl'))
cnn.eval()

# print 10 predictions from test data
inputs = test_x[:32]  # 测试32个数据
test_output = cnn(inputs)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')  # 打印识别后的数字
print(test_y[:32].numpy(), 'real number')

img = torchvision.utils.make_grid(inputs)
img = img.numpy().transpose(1, 2, 0)

# 下面三行为改变图片的亮度
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
cv2.imshow('win', img)  # opencv显示需要识别的数据图片
key_pressed = cv2.waitKey(0)