# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 22:06:03 2022

@author: naouf
"""

import torch.nn as nn
import torch

class TimeChangedCNF(nn.Module):
    

    def __init__(self, sequential_flow, time_change_func):
        
        super(TimeChangedCNF, self).__init__()
        
        self.sequential_flow = sequential_flow
        self.time_change_func = time_change_func

    def forward(self, x, logpx=None, reverse=False, inds=None):
        
        """
        reverse == True : x = aug_model(base_value, reverse = true)
        reverse == False : base_value = aug_model(x, reverse = false) 
        """
        x_hat = x[:,0]
        t_hat = x[:,1]
        
        
        phi_t = self.time_change_func(t_hat)
        
        
        x = torch.cat([x_hat[:,None], phi_t[:,None]], dim = 1 )
        
        x, logpx = self.sequential_flow(x, logpx, reverse, inds)
                
            
        return x, logpx