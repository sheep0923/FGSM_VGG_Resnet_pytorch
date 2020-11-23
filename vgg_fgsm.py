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

from attack_white import Attack

import time
 

#from PIL import Image
#from torchvision.utils import save_image

#导入VGG16模型
vggnet = models.vgg16(pretrained=False)
pthfile = r'/home/wangshouwen/data4/ljt/vgg16-397923af.pth'
#pthfile = r'vgg16-397923af.pth'
vggnet.load_state_dict(torch.load(pthfile))
#print(vgg16)



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

#损失函数定为交叉熵函数
criterion = nn.CrossEntropyLoss()

vggnet.eval()
attack = Attack(vggnet,criterion)


#遍历数据集，批量生成对抗样本
time_begin = time.time()
print("FGSM攻击vgg白盒攻击")
print("start:")
for eps in range(0,10):
    eps = (eps+1)/20
    acc_num = 0
    for j,data in enumerate(val_dataset_loader):
        inputs, labels = data
    
    #    print(j)
    
    #    print(inputs.size()) # out: (4, 3, 224, 224)
    #    print(labels) # out: tensor([0, 0, 0, 0])
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
            
        x = inputs
        y = labels
    #    print(x)
    #    print(y)
      
        img_adv,y_adv,y_pri = attack.fgsm(x,y,eps)
        _, y_adv = torch.max(y_adv.data, 1)
        _, y_pri = torch.max(y_pri.data, 1)
        
#        print(y_adv)
#        print(y_pri)
        for i in range(0,Batch_Size):
            if y_adv[i]==y_pri[i]:
                acc_num = acc_num + 1
                    
            
           
    acc_rate = acc_num/1000.0
    print("eps:" + str(eps) + "    acc_num:" + str(acc_num) + "    acc_rate:" + str(acc_rate))

print('finish!')
time_end = time.time()
time = time_end - time_begin
print('time:', time)

#    print(img_adv) 
#    print(img_adv.size())
#    print(img_adv)
#    if j == 0:
#        break




#    img=im_data.data
#    img-=img.min()
#    img/=img.max()
#    img*=255
#    img=img.cpu()
#    img=img.squeeze()
#    npimg=img.permute(1,2,0).numpy().astype('uint8')
#    plt.imsave('data/images/'+str(uuid1())+'.jpg',npimg)

