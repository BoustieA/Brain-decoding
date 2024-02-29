# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:14:17 2022

@author: adywi
"""
print("#"*10)
print("evaluation")

import os, fnmatch


import numpy as np
from utilities.Dataset import get_torch_datasets
from utilities.PATH import get_dic_path

from Models.DeepBrain_wang import DeepBrain
from Models.RNN_resnet18_depthwise import FEATURE_RNN_CF
from Models.CNN_LSTM_bis_ import FEATURE_RNN_CF
from Models.resnet18 import *
import torch
import torch.optim as optim
import torch.nn as nn
from Models.Attention_net_wang import Attention_network
import time

import matplotlib.pyplot as plt
import os


import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score

@torch.no_grad()
def get_prediction(model,dataset_loader,dtype,device):
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
        if i==0:
            print(x.shape)
            print(x.max())
            print(x.min())
        y    = batch[1].type(torch.LongTensor)
        y    = batch[1].to(device=device)
        Y = model(x)
        #print(Y)
        #loss_batch=criterion(Y, y)
        yhat = torch.argmax(Y,axis=-1)
        loss_batch+=0
        y_true+=y.tolist()
        y_pred+=yhat.tolist()
        t1 = time.time()

    print("loss:",loss_batch/i+1)
    return y_true, y_pred

def get_transition_matrix(y_true,y_pred,norm=False):
    labels_target = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]
    labels_orig=["IMA","ISO","ISS", "SPP","VMW","do","rep","rime","phono", "proso", "sem"]#["IMA","ISO","ISS", "SPP","VMW"]
    dic_count={i:0 for i in range(len(labels_target))}
    dic_transition={i:dict(dic_count) for i in range(len(labels_orig))}
    
    for i,j in enumerate(y_true):
        dic_transition[j][y_pred[i]]+=1
    df_=pd.DataFrame(np.zeros((len(labels_orig),len(labels_target))))
    df_.columns=labels_target
    df_.index=labels_orig
    
    for orig in dic_transition:
        for target in dic_transition[orig]:
            df_.iloc[orig,target]=dic_transition[orig][target]
    CM=df_
    if not norm:
        ax=sns.heatmap(CM,annot=True,fmt=".0f")        
        plt.title("Matrice de transition")
        plt.tight_layout()
        plt.show()
    else:
        for i in CM.index:
           sum_ = CM.loc[i,:].sum()
           if sum_>0:
               CM.loc[i,:]=CM.loc[i,:]/sum_
        CM=CM*100
        ax=sns.heatmap(CM,annot=True,fmt=".1f") 
        
        for t in ax.texts: t.set_text(t.get_text() + " %")
        plt.title("Matrice de transition normalisé")
        plt.tight_layout()
        plt.show()

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


t0 = time.time()

print(torch.cuda.is_available())
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(os.path.abspath(os.curdir))

#### PARAMETERS ####

param_data={"DATA":"INLANG",
                "processing":"oversample",
                "env":"silenus",
                "split":"all",
                "len_seq":14,
                "normalization":"global",#time
                "crop":False}


image_name_="INLANG_inference_from_resnet_final_oversample"#image to save
##### LOAD MODEL #####

#architecture
model = DeepBrain()
#model = FEATURE_RNN_CF()
model=S3ConvXFCResnet(27,7)
#model=Attention_network()

#weights
model_name='checkpoint_24.pth.tar'
#model_name='Attention_net_weights.pth.tar'
#model_name="DeepBrain_Global_norm_2_suite_suite_final_model.pth"


dic_path=get_dic_path()

model.load_state_dict(torch.load(dic_path["path_models"]+model_name,
                                 map_location=torch.device('cpu'))['state_dict'])

#model.load_model(path+model_name)
## Load Datasets



print("model_name")
print(model_name)


#get data
train_set,val_set=get_torch_datasets(**param_data)#/!\ train
test_loader = torch.utils.data.DataLoader(val_set,
                            batch_size=5,#batch_size=parameterization.get("batchsize", 3),
                            shuffle=True,
                            num_workers=0)

print("n_sample", len(val_set))

#prediction

y_true, y_pred = get_prediction(model,test_loader,dtype,device)

##### plot and print logs #####


print(np.unique(y_true),np.unique(y_pred))
get_transition_matrix(y_true,y_pred,True)
image_name=f"TM_{image_name_}_normalized.png"
path_image=os.path.join(dic_path["path_transition_matrix"],image_name)
plt.savefig(path_image)
plt.close()

get_transition_matrix(y_true,y_pred)
image_name=f"TM_{image_name_}.png"
path_image=os.path.join(dic_path["path_transition_matrix"],image_name)
plt.savefig(path_image)


##### compute metrics à froid #####

y_true=(np.zeros_like(y_true)+2).astype(int)# +2 car la base inlang est du langage

print("f1_w :",f1_score(y_true, y_pred,average="weighted"))
print("accuracy :",accuracy_score(y_true, y_pred))