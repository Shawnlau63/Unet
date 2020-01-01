import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import csv
from PIL import Image


data_transforms = transforms.Compose([transforms.ToTensor()])

DATA_PATH = r'D:\答辩\Unet答辩\Unetproject1\UNET'


class Mydata(Dataset):
    def __init__(self, path):
        super(Mydata, self).__init__()
        self.path = path
        self.list = []

        with open(self.path, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            firstrow = next(reader)
            for row in reader:
                self.list.append(row)


    def __len__(self):
        return len(self.list)


    def __getitem__(self, index):
        impath, lapath = self.list[index]

        img_path = os.path.join(DATA_PATH, impath)
        img = Image.open(img_path)
        img_data = data_transforms(img) - 0.5

        label_path = os.path.join(DATA_PATH, lapath)
        label = Image.open(label_path)
        label_data = data_transforms(label)[:1]

        # print(img.shape)
        # print(label.shape)

        return img_data, label_data

# if __name__ == '__main__':
#     data_path = r'D:\答辩\Unet答辩\Unetproject1\UNET\dev.csv'
#     data = Mydata(data_path)
#     print(data[0][1].shape)