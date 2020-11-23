# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:34:47 2020

@author: LV
"""


import os
import shutil  #用于移动文件
import numpy as np

filename = 'false1.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径

images = []
labels = []
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行读取数据
        if not lines:
            break
            pass
        p_tmp, E_tmp = [str(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        images.append(p_tmp)  # 添加新读取的数据
        labels.append(E_tmp)
        pass
#    images = np.array(images) # 将数据从list类型转换为array类型。
#    labels = np.array(labels)
    pass
#labels.sort()
#max=labels[len(labels)-1]
#min=labels[0]
#
#print(max)
#print(min)
print(images) 
print(len(images))
print(labels)
for i in range(0,len(images)):
#    tmp = images[i].split('\n')
#    images[i] = tmp[0].split('\t')
    shutil.move("G:/gd_project2/imagenet1000/" + labels[i]+'/'+images[i],"G:/gd_project2/imagenet1000/false")
#print(images)