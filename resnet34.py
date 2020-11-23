# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:19:04 2020

@author: LV
"""

import torch
import torchvision

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

#记录错误图片的txt文件
#false_name=open("false1.txt",'w+') 

resnet34 = models.resnet34(pretrained=False)
#pthfile = r'/home/wangshouwen/data4/ljt/resnet34-333f7ec4.pth'
pthfile = r'resnet34-333f7ec4.pth'
resnet34.load_state_dict(torch.load(pthfile))
#print(resnet34)

Batch_Size=4

data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])


val_dataset = torchvision.datasets.ImageFolder(root='imagenet1000_alltrue',transform=data_transform)
#val_dataset = torchvision.datasets.ImageFolder(root='/home/wangshouwen/data4/ljt/imagenet1000',transform=data_transform)
val_dataset_loader = DataLoader(val_dataset,batch_size=Batch_Size, shuffle=False)

#print(val_dataset.classes) # #根据分的文件夹的名字来确定的类别
#print(val_dataset.class_to_idx) # 按顺序为这些类别定义索引为0,1...
#print(val_dataset.imgs) #返回一个列表，它是从所有文件夹中得到的图片的路径以及其类别
#print(len(val_dataset)) # 数据集大小（图像数量） 
#print(val_dataset[0]) # 数据集第一张图像张量以及对应的标签，二维元组
#print(val_dataset[0][0]) # 数据集第一张图像的张量
#print(val_dataset[0][1]) # 数据集第一张图像的标签



use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    resnet34 = resnet34.cuda()


resnet34.eval()
acc_num = 0

count = 0

for j,data in enumerate(val_dataset_loader):
    
    inputs, labels = data
    print(j)
    
    if use_gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    
    #outputs为一个二维向量，行为1，列为1000
    outputs = resnet34(inputs)
    #print(outputs.shape)
    _, preds = torch.max(outputs, 1)
    #取出二维向量out中每一行的最大值及下标，pred为每行最大值的下标组成的列表。
    
    labels = labels.cpu()
    preds = preds.cpu()    
   #把tensor转化成list
    labels=labels.numpy()
    labels=labels.tolist()
    preds=preds.numpy()
    preds=preds.tolist()
    for i in range(0,Batch_Size):
        labels[i] = val_dataset.classes[labels[i]]
        labels[i] = int(labels[i])
        
        if preds[i] == labels[i]:
            acc_num = acc_num+1 
        #挑选出错误的图片，并将名称存到TXT文件中
#        else:
#            print(val_dataset.imgs[j*4+i][0])
#            str1 = val_dataset.imgs[j*4+i][0]
#            str1 = str1[-28:]
#            false_name.write(str1+' '+ str(labels[i]) +'\n')
#    print(preds)
#    print(labels)  
#    if j == 20:
#        break
    

    


#false_name.close()

print(acc_num)
acc = acc_num/1000 
print(acc)




 
