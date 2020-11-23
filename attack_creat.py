# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:45:57 2020

@author: LV
"""
 
import torch
from torch.autograd import Variable
import torch.nn as nn
from advertorch.utils import normalize_by_pnorm

#定义where函数
def where(cond, x, y):
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)



#def normalize_by_pnorm(x,p,small_constant=1e-6):
#    assert isinstance(p,float) or isinstance(p,int)
#    norm = _get_norm_batch(x,p)
#    norm = torch.max(norm,torch,ones_like(norm)*small_constant)
#    return batch_multiply(1./norm,x)



#对抗样本生成类
class Attack():
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def fgsm(self, x, y, eps, targeted=False,x_val_min=-1, x_val_max=1):
        
        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv)

        if targeted:
            cost = self.criterion(h_adv, y)
        else:
            cost = -self.criterion(h_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        
        adv = x_adv - x

        return x_adv, adv
        

    def i_fgsm(self, x, y, eps, iteration, targeted=False,  alpha=1, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv = self.net(x_adv)
            if targeted:
                cost = self.criterion(h_adv, y)
            else:
                cost = -self.criterion(h_adv, y)

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = where(x_adv > x+eps, x+eps, x_adv)
            x_adv = where(x_adv < x-eps, x-eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)
        
        adv = x_adv - x
        x_adv = x + adv
        
        return x_adv, adv
    
    
    def mi_fgsm(self,x, y, eps, nb_iter, decay_factor=1.,
                 clip_min=-1., clip_max=1., targeted=False):
        
        delta = torch.zeros_like(x)
        g = torch.zeros_like(x)

        delta = nn.Parameter(delta)

        for i in range(nb_iter):

            if delta.grad is not None:
                delta.grad.detach_()
                delta.grad.zero_()

            img_adv = x + delta
            outputs = self.net(img_adv)
            loss = self.criterion(outputs, y)
            if targeted:
                loss = -loss
            loss.backward()
            
            #g = decay_factor * g + delta.grad.data / torch.norm(delta.grad.data, p=1)
            g = decay_factor * g + normalize_by_pnorm(delta.grad.data, p=1)

            #delta.data += eps_iter * torch.sign(g)
            delta.data += eps / nb_iter * torch.sign(g)

            delta.data = torch.clamp( 
                delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(
                x + delta.data, min=clip_min, max=clip_max) - x

        x_adv = x + delta.data
        adv = delta.data
        return x_adv,adv
        