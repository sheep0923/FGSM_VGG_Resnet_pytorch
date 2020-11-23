# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:45:57 2020

@author: LV
"""
 
import torch
from torch.autograd import Variable
import torch.nn as nn


class MomentumIterativeAttack(Attack, LabelMixin):
   
    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, decay_factor=1.,
            eps_iter=0.01, clip_min=0., clip_max=1., targeted=False):
        """
        Create an instance of the MomentumIterativeAttack.

        :param predict: forward pass function.
        :param loss_fn: loss function.
        :param eps: maximum distortion.
        :param nb_iter: number of iterations
        :param decay_factor: momentum decay factor.
        :param eps_iter: attack step size.
        :param clip_min: mininum value per input dimension.
        :param clip_max: maximum value per input dimension.
        :param targeted: if the attack is targeted.
        """
        super(MomentumIterativeAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.decay_factor = decay_factor
        self.eps_iter = eps_iter
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        g = torch.zeros_like(x)

        delta = nn.Parameter(delta)

        for i in range(self.nb_iter):

            if delta.grad is not None:
                delta.grad.detach_()
                delta.grad.zero_()

            imgadv = x + delta
            outputs = self.predict(imgadv)
            loss = self.loss_fn(outputs, y)
            if self.targeted:
                loss = -loss
            loss.backward()

            g = self.decay_factor * g + normalize_by_pnorm(
                delta.grad.data, p=1)
            # according to the paper it should be .sum(), but in their
            #   implementations (both cleverhans and the link from the paper)
            #   it is .mean(), but actually it shouldn't matter

            delta.data += self.eps_iter * torch.sign(g)
            # delta.data += self.eps / self.nb_iter * torch.sign(g)

            delta.data = clamp(
                delta.data, min=-self.eps, max=self.eps)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = x + delta.data
        return rval