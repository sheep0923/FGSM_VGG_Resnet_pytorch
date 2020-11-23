# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:04:39 2020

@author: LV
"""

import torch
import torchvision
import torch.nn as nn

import numpy as np
import cv2


from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

from attack_creat import Attack
#from PIL import Image
#from torchvision.utils import save_image

#导入VGG16模型
vggnet = models.vgg16(pretrained=False)
#pthfile = r'/home/wangshouwen/data4/ljt/vgg16-397923af.pth'
pthfile = r'vgg16-397923af.pth'
vggnet.load_state_dict(torch.load(pthfile))
#print(vgg16)


#对图片的预处理
data_transform = transforms.Compose([
#        transforms.Resize(256),
#        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

 
trans_compose = transforms.Compose(data_transform)   
    
#导入imagenet数据集
Batch_Size = 1
#val_dataset = torchvision.datasets.ImageFolder(root='/home/wangshouwen/data4/ljt/imagenet1000_alltrue',transform=data_transform)
val_dataset = torchvision.datasets.ImageFolder(root='imagenet1000_alltrue',transform=data_transform)
val_dataset_loader = DataLoader(val_dataset,batch_size=Batch_Size, shuffle=False)   


use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    vggnet = vggnet.cuda()

#损失函数定为交叉熵函数
criterion = nn.CrossEntropyLoss()

vggnet.eval()
attack = Attack(vggnet,criterion)


#遍历数据集，批量生成对抗样本

print("start:")
acc_num = 1
eps=0.3
ite = 4

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


for j,data in enumerate(val_dataset_loader):
    inputs, labels = data
    
    print(j)
    
    #    print(inputs.size()) # out: (4, 3, 224, 224)
    #    print(labels) # out: tensor([0, 0, 0, 0])
    if use_gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
        
    x = inputs
    y = labels
    
#    img_adv,adv = attack.fgsm(x,y,eps)
#    img_adv,adv= attack.i_fgsm(x,y,eps,ite)
    img_adv,adv= attack.mi_fgsm(x,y,eps,ite)
#    print(img_adv)
#    print(adv)
#    print(x)

#        print(y_adv)
#        print(y_pri)
#    print(img_adv.size())

    
    for i in range(0,Batch_Size):
        
        x = img_adv[i]
        x[0]=x[0]*std[0]+mean[0]
        x[1]=x[1]*std[1]+mean[1]
        x[2]=x[2].mul(std[2])+mean[2]
        img = x.mul(255).byte()
        img = img.numpy().transpose((1, 2, 0))
        #img = img.numpy()
        #torch.set_num_threads(3)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #print(img)
        #cv2.imshow("sdf", img)
        img_path='G:\gd_project2\img_adv\MIM' + str(j)+'eps='+str(eps) +'.jpg'
        #img_path='G:\gd_project2\img_adv\MIM' + str(j)+'ite='+str(ite) +'.jpg'
        #img_path='G:\gd_project2\img_adv\BIM' + str(j)+'eps='+str(eps) +'.jpg'
        #img_path='G:\gd_project2\img_adv\BIM' + str(j) + 'ite='+str(ite) +'.jpg'
        #img_path='G:\gd_project2\img_adv\FGSM' + str(j)+'eps='+str(eps) +'.jpg'
        cv2.imwrite(img_path,img)
        
#        adv = adv[i]
#        adv[0]=adv[0]*std[0]+mean[0]
#        adv[1]=adv[1]*std[1]+mean[1]
#        adv[2]=adv[2].mul(std[2])+mean[2]
#        adv = adv.mul(255).byte()
#        adv = adv.numpy().transpose((1, 2, 0))
#        #img = img.numpy()
#        #torch.set_num_threads(3)
#        adv=cv2.cvtColor(adv,cv2.COLOR_BGR2RGB)
#        #cv2.imshow("sdf", img)
#        adv_path='G:\gd_project2\img_adv\adv' + str(j) +'.jpg'
#        cv2.imwrite(adv_path,adv)        
        
      
    if j == 5:
        break
    
print('finish!')



#    img=im_data.data
#    img-=img.min()
#    img/=img.max()
#    img*=255
#    img=img.cpu()
#    img=img.squeeze()
#    npimg=img.permute(1,2,0).numpy().astype('uint8')
#    plt.imsave('data/images/'+str(uuid1())+'.jpg',npimg)

