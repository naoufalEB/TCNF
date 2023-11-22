# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:43:49 2022

@author: naouf
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .odefunc import NONLINEARITIES

#f_activ = NONLINEARITIES['softplus']
f_activ = nn.Softplus()


class time_change_func_id(nn.Module):
    def __init__(self):
        super(time_change_func_id, self).__init__()
        
                
    def forward(self, t):
                
        phi_t = t
                
        return phi_t
        
    
class time_change_func_exp(nn.Module):
    def __init__(self):
        super(time_change_func_exp, self).__init__()
        
        self.theta = nn.Parameter(torch.rand(1))
        
    def forward(self, t):
                
        phi_t = torch.exp(2*torch.abs(self.theta)*t) - 1
                
        return phi_t
        

# anti-derivative of Relu
def primitive_Relu(t):
    s_k = 0.5*(t**2)*(t>=0) + 0*(t<0)
    res = torch.sum(s_k, dim = 0)
    return res

def positif_phi_t(t, eps = 0.01):
    
    min_t,_ = torch.min(t, 0)
    
    correction = ( eps - min_t )*( min_t <= 0 )
    
    res = t + correction
        
    return res

class time_change_func_M_MGN(nn.Module):
    def __init__(self, K, N, dim_input = 1):
        super(time_change_func_M_MGN, self).__init__()
        
        dim_input
        vec_b = []
        vec_W = []
              
        
        for i in range(K):
            b_temp = nn.Parameter(torch.rand(N, 1))
            vec_b.append(b_temp)
            
            w_temp = nn.Parameter(torch.rand(N, dim_input))
            vec_W.append(w_temp)
            
        self.b_array = nn.ParameterList(vec_b)
        self.W_array = nn.ParameterList(vec_W)
        
        self.V = nn.Parameter(torch.eye(N, dim_input))
        
        self.a = nn.Parameter(torch.zeros(dim_input,1))
        
        self.activation_fn1 = nn.ReLU() 

    def forward(self, t):
        
        t1 = t[None,:]
        K = len(self.b_array)
        
        NN_module_sum = 0
        for i in range(K):
            temp_z_i = torch.matmul(self.W_array[i], t1) + self.b_array[i]
            
            NN_module_i = primitive_Relu(temp_z_i)*(torch.matmul(torch.t(self.W_array[i]), self.activation_fn1(temp_z_i)))
            
            NN_module_sum = NN_module_sum + NN_module_i
        
        cross_prod = torch.matmul( torch.matmul(torch.t(self.V), self.V), t1)
        
        MGN = torch.reshape(self.a + cross_prod + NN_module_sum, t.shape)
        
        MGN = positif_phi_t(MGN)
        
        return MGN