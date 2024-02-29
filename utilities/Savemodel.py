#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:59:13 2022

@author: neurodeep
"""
import os
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, save_parameters, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.save_parameters = save_parameters
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion, scheduler
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            dic_save=dict(self.save_parameters)
            dic_save["model_name"]+="_best_model"
            save_model(epoch, model, optimizer, criterion, scheduler, dic_save)
            
def save_model(epochs, model, optimizer, criterion, scheduler, parameters):
    model_name = parameters.get("model_name","final_model")
    path_records = parameters.get("path_models","Records/trained_model")
    path_save = os.path.join(path_records, model_name)+".pth"
    """
    Function to save the trained model to disk.
    """
    #print("Saving model...")
    torch.save({
                'epoch': epochs,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'state_dict': model.state_dict(),
                'loss': criterion,
                }, path_save)