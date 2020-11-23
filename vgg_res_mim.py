# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:04:39 2020

@author: LV
"""


import torch
import torchvision
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

from attack_black import Attack_Black
#from PIL import Image
#from torchvision.utils import save_image

#导入VGG16模型
vggnet = models.vgg16(pretrained=False)
pthfile = r'/home/wangshouwen/data4/ljt/vgg16-397923af.pth'
#pthfile = r'vgg16-397923af.pth'
vggnet.load_state_dict(torch.load(pthfile))
#print(vgg16)

#导入Resnet34模型
resnet34 = models.resnet34(pretrained=False)
pthfile = r'/home/wangshouwen/data4/ljt/resnet34-333f7ec4.pth'
#pthfile = r'resnet34-333f7ec4.pth'
resnet34.load_state_dict(torch.load(pthfile))
#print(resnet34)

#对图片的预处理
data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

#导入imagenet数据集
Batch_Size = 4
val_dataset = torchvision.datasets.ImageFolder(root='/home/wangshouwen/data4/ljt/imagenet1000_alltrue',transform=data_transform)
#val_dataset = torchvision.datasets.ImageFolder(root='imagenet1000_alltrue',transform=data_transform)
val_dataset_loader = DataLoader(val_dataset,batch_size=Batch_Size, shuffle=False)   


use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    vggnet = vggnet.cuda()
    resnet34 = resnet34.cuda()

#损失函数定为交叉熵函数
criterion = nn.CrossEntropyLoss()

vggnet.eval()
resnet34.eval()
attack = Attack_Black(vggnet,resnet34,criterion)


#遍历数据集，批量生成对抗样本
print("MIM__vgg攻击resnet")
print("start:")

#超参数的设置
for ite in range(0,10):
#for eps in range(0,10):
    acc_num = 0

    eps = 0.3
    iteration = (ite+1)*2
#    eps = (eps+1)/10
#    iteration = 10
    

    for j,data in enumerate(val_dataset_loader):
        inputs, labels = data
        
            #print(j)
        
            #print(inputs.size()) # out: (4, 3, 224, 224)
            #print(labels) # out: tensor([0, 0, 0, 0])
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
                
        x = inputs
        y = labels
            #print(x)
            #print(y)

        img_adv,y_adv,y_pri = attack.mi_fgsm(x,y,eps,iteration)
        _, y_adv = torch.max(y_adv.data, 1)
        _, y_pri = torch.max(y_pri.data, 1)
        
#        print(y_adv)
#        print(y_pri)
        for i in range(0,Batch_Size):
            if y_adv[i]==y_pri[i]:
                acc_num = acc_num + 1

                
               
    acc_rate = acc_num/1000.0
    print("eps:" + str(eps) +"    iteration:"+ str(iteration) + "    acc_num:" + str(acc_num) + "    acc_rate:" + str(acc_rate))
print('finish!')
