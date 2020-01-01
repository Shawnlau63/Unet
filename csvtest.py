import os
import csv

li = []
path = r'E:\AI\UNET\dev.csv'
with open(path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = next(reader)
    for row in reader:
        li.append(row)
print(rows)
print(len(li))

img_path, label = li[0]

print(img_path, label)
