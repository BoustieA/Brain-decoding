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
import gc
from Models.DeepBrain import DeepBrain
from Models.RNN_resnet18_depthwise import FEATURE_RNN_CF
from Models.CNN_LSTM_bis_ import FEATURE_RNN_CF
from Models.resnet18 import S3ConvXFCResnet
from Models.Attention_net_wang import Attention_network
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import time

import matplotlib.pyplot as plt

import torch
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,balanced_accuracy_score
import seaborn as sns
import pandas as pd


@torch.no_grad()
def get_prediction(model,dataset_loader,dtype,device):
    t0 = time.time()
    y_pred=[]
    y_true=[]
    torch.no_grad()
    model.eval()
    model.to(device)
    for i,batch in enumerate(iter(dataset_loader)):
        x    = batch[0].to(dtype=dtype, device=device)
        y    = batch[1].type(torch.LongTensor)
        y    = batch[1].to(device=device)
        x = model.Feature_extractor(x)
        #x=torch.cat([x,y[:,None].to(device=device)//2],axis=-1)
        Y=model.classifier(x)
        #print(Y)
        #yhat = torch.argmax(Y,axis=-1)
        
        y_true+=(y).tolist()
        y_pred+=Y.tolist()
    
    return y_true, y_pred



def plot_confusion_matrix(y_true, y_pred,normalize=None):
    #plt.figure(figsize=(6,7))
    #labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]
    labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]
    #labels=["IMA","ISO","ISS", "SPP","VMW"]
    labels=["MONITORING","SEMANTIC","WANDERING", "PRODUCTION", "DECODING"]
    labels=["task","control"]
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


t0 = time.time()
print(torch.cuda.is_available())
dtype=torch.float
torch.backends.cudnn.benchmark=True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(os.path.abspath(os.curdir))


##### PARAMETERS #####
image_name_="resnet TC freezed phono"#name of the image
model_="resnet"#architecture to load

param_data={"DATA":"INLANG_CONTROL",
                "env":"silenus",
                "split":"final_test",
                "crop":False}#train_val_test #test#"final_test"
classifier_param={'drop_out2': 0.1,
                      'drop_out1': 0.4116061583702507,
                      'embedding': 'False',
                      "activation":"ELU",
                      "layer_size":128*2,
                      "num_class":2,
                      "range_init":0.1}

#classifier
CF=nn.Sequential(#nn.Dropout(0.5),
    nn.Linear(512,2))


# get some path

dic_path=get_dic_path()

##### SELECT MODEL #####

if model_=="attention":
    ## Ne marche pas tel quel, il faut redéfinir le classifier sous la forme .fc
    model=Attention_network()
    classifier_param["input_size"]=128
    model_name="Finetuning_Resnet_INLANG_Task_vs_Control.pth"#"Resnet_INLANG_GS_final_model.pth"
    D=torch.load(dic_path["path_models"]+model_name)["state_dict"]
    model.load_state_dict(D)
    model.fc=CF

elif model_=="resnet":
    model=S3ConvXFCResnet(27,7)
    param_data["normalization"]="global"
    param_data["processing"]="oversample"
    param_data["len_seq"]=14
    classifier_param["input_size"]=512
    
    model.classifier=CF



##### Load Datasets #####

train_set,val_set=get_torch_datasets(**param_data)#/!\ train
val_set=train_set
val_set.get_weights_labels()


##########




print("n_sample", len(val_set))

test_loader = torch.utils.data.DataLoader(val_set,
                            batch_size=5,#batch_size=parameterization.get("batchsize", 3),
                            shuffle=False,
                            num_workers=0)




### List of model to load ###


#list_model = [f'Finetuning_Resnet_INLANG_Task_vs_Control_sem_{i}' for i in range(10)]    
#list_model = [f'Finetuning_Resnet_INLANG_5_class_2_{i}' for i in range(8)]    
list_model = [f'Finetuning_Resnet_INLANG_Task_vs_Control_phono_smoothed_{i}' for i in range(10)]

#### change classifier ####


y_pred_list=[]
W=[0.825,0.9,0.95,0.625,1,0.825,0.725,0.875,0.7,1]#weights of the models
W=np.array(W)
for model_name in list_model:#for each model : load and get prediction
    D=torch.load(dic_path["path_models"]+model_name+"_best_model.pth")["state_dict"]
    model.load_state_dict(D)
    y_true, y_pred = get_prediction(model,test_loader,dtype,device)
    y_pred_list+=[y_pred]
y_pred_list=np.array(y_pred_list)

#get_score for each model
f1_score_=[]
acc_score_=[]
for i in range(len(list_model)):
    y_pred=y_pred_list[i]
    y_pred=np.argmax(y_pred,axis=-1)
    f1_score_+=[f1_score(y_true, y_pred,average="weighted")]
    acc_score_+=[balanced_accuracy_score(y_true,y_pred)]
print("f1 score list:",f1_score_)
print("acc_w score list:",acc_score_)

#weightened average of predictions
y_pred_list=y_pred_list*W[:,None,None]#weightened the prediction according to model score
print("shape of prediction")
print(y_pred_list.shape)
y_pred_prob=y_pred_list.sum(axis=0)/np.sum(W)
y_pred=np.argmax(y_pred_prob,axis=-1)





#compute metrics
print("@"*5,"\nSCORES:")
print("mean model")
print("f1_w :",f1_score(y_true, y_pred,average="weighted"))
print("acc_w :",balanced_accuracy_score(y_true,y_pred))
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


print("y_true")
print(y_true)

