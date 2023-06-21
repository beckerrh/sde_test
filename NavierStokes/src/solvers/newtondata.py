#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:16 2016

@author: becker
"""

import numpy as np

# class IterationInfo:
#     def __init__(self, niter, nliniter, success=True, failure=""):
#         self.success, self.failure = success, failure
#         self.niter, self.nliniter = niter, nliniter

class StoppingParamaters:
    def __init__(self, **kwargs):
        self.maxiter = kwargs.pop('maxiter',100)
        self.atol = kwargs.pop('atol',1e-12)
        self.rtol = kwargs.pop('rtol',1e-8)
        self.atoldx = kwargs.pop('atoldx',1e-12)
        self.rtoldx = kwargs.pop('rtoldx',1e-8)
        self.divx = kwargs.pop('divx',1e8)
        self.rho_aimed = kwargs.pop('rho_aimed',0.1)
        self.firststep = 1.0
        self.steptype = kwargs.pop('steptype','backtracking')
        if 'nbase' in kwargs: self.nbase = kwargs.pop('nbase')
        self.bt_maxiter = kwargs.pop('bt_maxiter',10)
        self.bt_omega = kwargs.pop('bt_omega',0.75)
        self.bt_c = kwargs.pop('bt_c',0.1)
        self.maxter_stepsize = 5

class IterationData:
    def __repr__(self):
        all = [f"{k}: {v}" for k,v in self.__dict__.items()]
        return ' '.join(all)
    def __init__(self, resnorm, **kwargs):
        self.calls = 0
        self.totaliter, self.totalliniter = 0, 0
        self.reset(resnorm)
        self.calls = 0
        self.bad_convergence_count = 0
    def niter_mean(self):
        if not self.calls: return 1
        return self.totaliter/self.calls
    def niter_lin_mean(self):
        return np.mean(np.array(self.liniter))
    def reset(self, resnorm):
        self.calls += 1
        if hasattr(self, 'iter'): self.totaliter += self.iter
        if hasattr(self, 'liniter'): self.totalliniter += np.sum(self.liniter)
        self.liniter, self.dxnorm, self.resnorm, self.step = [], [], [], []
        self.iter = 0
        self.success = True
        self.resnorm.append(resnorm)
    def newstep(self, dx, liniter, resnorm, step):
        self.liniter.append(liniter)
        self.dxnorm.append(np.linalg.norm(dx))
        self.resnorm.append(resnorm)
        self.step.append(step)
        if len(self.dxnorm)>1:
            self.rhodx = self.dxnorm[-1]/self.dxnorm[-2]
        else:
            self.rhodx = 0
        self.rhor = self.resnorm[-1]/self.resnorm[-2]
        self.iter += 1
       