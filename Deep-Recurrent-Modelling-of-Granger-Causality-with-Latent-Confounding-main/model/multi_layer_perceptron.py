# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:29:41 2021

@author: tonyz
"""

import torch.nn as nn


def one_layer(input_size,output_size,activation,layer_type=None, dropout=None):
    
    if activation == "relu":
        act = nn.ReLU(inplace=True)
    if activation == "tanh":
        act = nn.Tanh()
    if activation == "sigmoid":
        act = nn.Sigmoid()
    layer=[]
    if layer_type is None:
        layer.append(nn.Linear(input_size,output_size))
        layer.append(act)
        if dropout is not None:
            layer.append(dropout)
    else:
        layer.append(nn.Linear(input_size,output_size))
        
    return nn.Sequential(*layer)


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, layers, activation='relu', dropout=None):

        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        sequence = []
        for layer in range(self.layers-1):
            if layer == 0:
                sequence.append(one_layer(self.input_size,self.hidden_size,activation))
            elif layer == self.layers-2:
                sequence.append(one_layer(self.hidden_size,self.output_size,activation,"output"))
            else:
                sequence.append(one_layer(self.hidden_size,self.hidden_size,activation, self.dropout))
        
        self.mlp = nn.Sequential(*sequence)
        self.layers = nn.ModuleList(sequence)
        
    def forward(self, x):

        return self.mlp(x)

