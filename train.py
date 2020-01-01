import os
import torch
import torch.optim as optim
from mydata import Mydata
from net import Unet
from torch.utils import data
import matplotlib.pyplot as plt


class Train():
    def __init__(self, save_path, data_path):
        self.net = Unet()
        self.save_path = save_path
        self.data_path = data_path

        # 判断设备是否支持cuda，若支持，则使用cuda训练
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = optim.Adam(self.net.parameters())

        if os.path.exists(self.save_path):
            self.net = torch.load(self.save_path)
            print('模型已存在，继续训练')
        else:
            print('模型不存在，训练新的')

        # 网络是训练模式
        self.net.train()
        self.net.to(self.device)

    def train(self):
        my_data = Mydata(self.data_path)
        train_data = data.DataLoader(dataset=my_data, batch_size=5, shuffle=True)

        losses = []

        epoch = 0

        while True:
            # 循环取出数据和标签，进行训练
            for i, (img_data, label_data) in enumerate(train_data):
                img_data = img_data.to(self.device)
                label_data = label_data.to(self.device)

                # 将数据放入网络，获取结果
                output = self.net(img_data)

                loss = torch.sum((output - label_data) ** 2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.float())
                print('epoch - {} - {}/{} loss: {}'.format(epoch, i, len(train_data), loss.float()))

            epoch += 1

            torch.save(self.net, self.save_path)

            if len(losses) > 5000:
                losses = losses[-5000:]

            plt.clf()
            plt.plot(losses)
            plt.savefig('Unet_loss.jpg')