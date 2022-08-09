"""一个GAN的简单test"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
train_ds = torchvision.datasets.MNIST('data',
                                      train=True,
                                      transform=transform,
                                      download=False)

dataloader = torch.utils.data.DataLoader(
    train_ds, batch_size=128, shuffle=True)


# 定义生成器
# 输入是长度为100的噪声
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gan = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            # 最后生成一张28*28的图品
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.gan(x)
        img = img.view(-1, 28, 28)
        return img


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(nn.Linear(28*28, 512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 256),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.disc(x)
        return x


# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gan = Generator().to(device)
dis = Discriminator().to(device)

# 定义优化器
d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(gan.parameters(), lr=0.0001)

# 损失函数
loss_function = torch.nn.BCELoss()


# 绘图函数
def gen_img_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axis('off')
    plt.show()


test_input = torch.randn(16, 100, device=device)

# 开始训练
D_loss = []
G_loss = []

for epoch in range(50):
    d_epoch_loss = 0
    g_epoch_loss = 0
    batch_count = len(dataloader.dataset)

    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)

        # 随机生成的图片要与真实图片维度相同
        random_noise = torch.randn(size, 100, device=device)
        d_optim.zero_grad()

        # 判别式前向传播 判别真实图片
        real_output = dis(img)
        # 判别式算是
        d_read_loss = loss_function(real_output,
                                    torch.ones_like(real_output))
        # 判别式反向传播 不进行梯度更新
        d_read_loss.backward()



        # 生成器前向传播
        gen_img = gan(random_noise)
        # 判别 假图片的真实性
        fake_output = dis(gen_img.detach())
        # 假图片判别生假图片的损失
        d_fake_loss = loss_function(fake_output,
                                    torch.zeros_like(fake_output))
        d_fake_loss.backward()

        d_loss = d_read_loss + d_fake_loss

        d_optim.step()




        # 得到生成器的损失
        g_optim.zero_grad()
        fake_output = dis(gen_img)
        # 生成假图片是真图片的损失
        g_loss = loss_function(fake_output,
                               torch.ones_like(fake_output))
        g_loss.backward()
        g_optim.step()

        # print(d_loss)
        # print(g_loss)
        # exit()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    with torch.no_grad():
        d_epoch_loss /= batch_count
        g_epoch_loss /= batch_count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch', epoch)
        gen_img_plot(gan, test_input)



