# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:45:57 2020

@author: LV
"""
 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from utils import clamp
from utils import normalize_by_pnorm

from utils import batch_multiply

def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


#def batch_multiply(float_or_vector, tensor):
#    if isinstance(float_or_vector, torch.Tensor):
#        assert len(float_or_vector) == len(tensor)
#        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
#    elif isinstance(float_or_vector, float):
#        tensor *= float_or_vector
#    else:
#        raise TypeError("Value has to be float or torch.Tensor")
#    return tensor
#
#
#def normalize_by_pnorm(x, p=2, small_constant=1e-6):
#    # loss is averaged over the batch so need to multiply the batch
#    # size to find the actual gradient of each input sample
#
#    assert isinstance(p, float) or isinstance(p, int)
#    norm = _get_norm_batch(x, p)
#    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
#    return batch_multiply(1. / norm, x)



def replicate_input(x):
    return x.detach().clone()

def _verify_and_process_inputs(x, y):
    x = replicate_input(x)
    y = replicate_input(y)
    return x, y


#定义where函数
def where(cond, x, y):
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


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

        x_adv = x_adv - eps*torch.sign(x_adv.grad)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)


        h = self.net(x)
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h
        

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


        h = self.net(x)
        h_adv = self.net(x_adv)
        
        return x_adv, h_adv, h


#    def mi_fgsm(self,x, y, eps, iteration, decay_factor=1., targeted=False, x_val_min=-1, x_val_max=1):  
#    
#        x_adv = Variable(x.data, requires_grad=True)
#        g=0
#        for i in range(iteration):
#            h_adv = self.net(x_adv)
#            if targeted:
#                cost = self.criterion(h_adv, y)
#            else:
#                cost = -self.criterion(h_adv, y)
#
#            self.net.zero_grad()
#            if x_adv.grad is not None:
#                x_adv.grad.data.fill_(0)
#            cost.backward()
#
#
#            g = decay_factor * g + x_adv.grad.data / torch.norm(x_adv.grad.data, p=1)
#            #g = decay_factor * g + normalize_by_pnorm(x_adv.grad.data, p=1)
#            
#            #x_adv.data = x_adv.data + torch.sign(g)            
#            x_adv.data = x_adv.data + eps / iteration * torch.sign(g)
#
#            x_adv = where(x_adv > x+eps, x+eps, x_adv)
#            x_adv = where(x_adv < x-eps, x-eps, x_adv)
#            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
#            x_adv = Variable(x_adv.data, requires_grad=True)
#            
#        h = self.net(x)
#        h_adv = self.net(x_adv)
#        
#        return x_adv, h_adv, h


    def mi_fgsm(self,x, y, eps, nb_iter, decay_factor=1., targeted=False, clip_min=-1, clip_max=1):  
             
             
        x, y = _verify_and_process_inputs(x, y)
        delta = torch.zeros_like(x)
        g = torch.zeros_like(x)

        delta = nn.Parameter(delta)

        for i in range(nb_iter):

            if delta.grad is not None:
                delta.grad.detach_()
                delta.grad.zero_()

            imgadv = x + delta
            outputs = self.net(imgadv)
            loss = self.criterion(outputs, y)
            if targeted:
                loss = -loss
            loss.backward()
    
            g = decay_factor * g + normalize_by_pnorm(
                delta.grad.data, p=1)
            #delta.data += torch.sign(g)
            delta.data += torch.sign(g)
            delta.data = clamp(
                    delta.data, min=-eps, max=eps)
            delta.data = clamp(
                    x + delta.data, min=clip_min, max=clip_max) - x
        
        
        rval = x + delta.data
        h = self.net(x)
        h_adv = self.net(rval)
        
        return rval,h,h_adv



