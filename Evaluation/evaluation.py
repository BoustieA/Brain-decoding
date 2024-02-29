# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:14:17 2022

@author: adywi
"""
print("#"*10)
print("evaluation")

import os, fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from utilities.Dataset import get_torch_datasets
from utilities.PATH import get_dic_path
from utilities.SPLIT import *

from Models.DeepBrain import DeepBrain
from Models.RNN_resnet18_depthwise import FEATURE_RNN_CF
from Models.CNN_LSTM_bis_ import FEATURE_RNN_CF
from Models.resnet18 import S3ConvXFCResnet
from Models.Attention_net_wang import Attention_network
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score







t0 = time.time()
print(torch.cuda.is_available())
dtype=torch.float
torch.backends.cudnn.benchmark=True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(os.path.abspath(os.curdir))

### PARAMETERS ###

image_name_="DB_test"#name of the images

model_="resnet"#model to load "resnet","attention","DeepBrain"


param_data={"DATA":"HCP",
            "env":"silenus",
            "split":"test",
            "crop":False}


### charge quelques chemins###
dic_path=get_dic_path()



##### LOAD MODEL #####

if model_=="attention":
    model=Attention_network()
    w_load="Attention_net_weights.pth.tar"#
    D=torch.load("./"+dic_path["path_models"]+w_load)["state_dict"]
    model.load_state_dict(D)
    input_size=128
    classifier_name="Finetuning_Attention_INLANG_best_model.pth"

elif model_=="resnet":
    model=S3ConvXFCResnet(27,7)

    w_load="DeepBrain_Global_norm_2_suite_suite_final_model.pth"
    D=torch.load("./"+dic_path["path_models"]+w_load)["state_dict"]
    model.load_state_dict(D)
    #model=model.Feature_extractor
    param_data["normalization"]="global"
    param_data["processing"]="raw"#"oversample"
    param_data["len_seq"]=27
    input_size=512
    classifier_name="Resnet_INLANG_GS_best_model.pth"#"Resnet_INLANG_GS_final_model.pth"

elif model_=="DeepBrain":
    model=DeepBrain()
    w_load="checkpoint_24.pth.tar"
    D=torch.load("./"+dic_path["path_models"]+w_load)["state_dict"]
    #model.load_state_dict(D)
    model.load_model("./"+dic_path["path_models"]+w_load)
    #model=model.Feature_extractor
    param_data["normalization"]="time"
    param_data["processing"]="raw"
    param_data["len_seq"]=27
    input_size=512
    classifier_name="Resnet_INLANG_GS_best_model.pth"#"Resnet_INLANG_GS_final_model.pth"





### LOAD dataset ###
train_set,val_set = get_torch_datasets(**param_data)
print("n_sample", len(val_set))

test_loader = torch.utils.data.DataLoader(val_set,
                            batch_size=5,#batch_size=parameterization.get("batchsize", 3),
                            shuffle=True,
                            num_workers=0)





@torch.no_grad()
def get_prediction(model,dataset_loader,dtype,device):
    """
    récupère la classe prédite par le modèle pour les échantillons du dataset
    """
    t0 = time.time()
    y_pred=[]
    y_true=[]
    torch.no_grad()
    model.eval()
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    loss_batch=0
    for i,batch in enumerate(dataset_loader):
        x    = batch[0].to(dtype=dtype, device=device)
        y    = batch[1].type(torch.LongTensor)
        y    = batch[1].to(device=device)
        Y = model(x)
        loss_batch+=criterion(Y, y)*x.shape[0]
        yhat = torch.argmax(Y,axis=-1)
        y_true+=y.detach().cpu().tolist()
        y_pred+=yhat.detach().cpu().tolist()
        t1 = time.time()
    loss=loss_batch/len(dataset_loader.dataset)
    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred,normalize=None):
    """
    plot la matrice de confusion, labels à définir selon les besoins
    """
    #plt.figure(figsize=(8,8))
    #labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]
    labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]
    #labels=["IMA","ISO","ISS", "SPP","VMW"]
    #labels=["MONITORING","SEMANTIC","WANDERING", "PRODUTION", "DECODING"]
    #labels=["task","control"]
    if normalize:
        CM=confusion_matrix(y_true, y_pred, normalize=normalize)*100
        CM=pd.DataFrame(CM,columns=labels)
        CM.index=labels
        ax=sns.heatmap(CM,annot=True,fmt=".1f")
        for t in ax.texts: t.set_text(t.get_text() + " %")
        ax.set_xlabel("Predictions")
        ax.set_ylabel("Vrai labels")
    else:
        CM=confusion_matrix(y_true, y_pred)
        CM=pd.DataFrame(CM,columns=labels)
        sns.heatmap(CM,annot=True)
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

#prediction
y_true, y_pred = get_prediction(model,test_loader,dtype,device) 
#np.save(dic_path["path_confusion_matrix"]+"y_true_resnet18.npy",y_true) # sauvegarde si besoin, pour 
#np.save(dic_path["path_confusion_matrix"]+"y_pred_resnet18.npy",y_pred) # 

#print les scores dans les logs
print("@"*5,"\nSCORE :")
print("f1_w :",f1_score(y_true, y_pred,average="weighted"))
print("accuracy :",accuracy_score(y_true, y_pred))
print(np.unique(y_true),np.unique(y_pred))
plot_confusion_matrix(y_true, y_pred,normalize="true")
image_name=f"CM_{image_name_}_normalized.png"
path_image=os.path.join(dic_path["path_confusion_matrix"],image_name)
plt.savefig(path_image)
plt.close()

plot_confusion_matrix(y_true, y_pred)
image_name=f"CM_{image_name_}.png"
path_image=os.path.join(dic_path["path_confusion_matrix"],image_name)
plt.savefig(path_image)



