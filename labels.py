# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 21:07:12 2020

@author: LV
"""
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
    images = np.array(images) # 将数据从list类型转换为array类型。
    labels = np.array(labels)
    pass

print(images[0])
print(labels[0])
