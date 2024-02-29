# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:45:03 2023

@author: 33695
"""


def get_n_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
