# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:34:47 2020

@author: LV
"""

#
#代码功能：将文件夹中的图片，分别放入到1000个类别中的小文件夹中
#


import pandas as pd
import os
import shutil  #用于移动文件
import numpy as np

filename = 'val.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
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
for i in range(1000):
    os.mkdir(str(i))

for i in range(0,50000):
    j=str(labels[i])
    shutil.move(images[i],j)
