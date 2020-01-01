import os
import torch
import torch.nn as nn
from mydata import Mydata
from torch.utils.data import DataLoader
import torch.optim as optim
from net import Unet

DATA_PATH = r'D:\答辩\Unet答辩\Unetproject1\UNET\dev.csv'
SAVE_PATH = './params/unet.pkl'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

net = Unet()
opt = optim.Adam(net.parameters())
loss_fun = nn.BCELoss()

if not os.path.exists('./params'):
    os.mkdir('./params')

if os.path.exists(SAVE_PATH):
    print('模型已存在，继续训练！')
    net.load_state_dict(torch.load(SAVE_PATH))



train_data = Mydata(DATA_PATH)
dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

for epoch in range(10000):
    for i, (img, label) in enumerate(dataloader):
        img = img
        label = label

        target = net(img)

        loss = loss_fun(target, label)

        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch % 5 == 0:
        torch.save(net.state_dict(), SAVE_PATH)
    print('epoch:{} loss:{}'.format(epoch, loss.item()))