# -*- coding: utf-8 -*-


#Utilities
import os, fnmatch
import time
import torch
import numpy as np
import dill as pickle

from utilities.Savemodel import save_model
from utilities.Dataset import BrainDataset,get_torch_datasets
from utilities.PATH import  get_dic_path,get_subject_ID

import matplotlib.pyplot as plt

#Model and training
from Models.DeepBrain import DeepBrain
from Models.resnet18 import S3ConvXFCResnet
#from Models.RNN_resnet18_depthwise_wide import FEATURE_RNN_CF
from Models.CNN_LSTM import FEATURE_RNN_CF2 as FEATURE_RNN_CF
from Train.train_func_scheduler_FT import Training
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if __name__=="__main__":
    torch.backends.cudnn.benchmark=True
    print("###"*10)
    print("START SCRIPT")
    print("###"*10)
    t0 = time.time()
    torch.cuda.empty_cache()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("#"*10)
    n_device=torch.cuda.device_count()
    print(f"Training on {n_device} {device}")
    print()
    param_data={"DATA":"PHONO_CONTROL_SMOOTHED",
            "processing":"oversample",
            "split":"all",
            "env":"bettik",
            "len_seq":27,
            "normalization":"global",
            "crop":False}
    #model to use as feature extractor
    model_architecture="resnet18"#"RNN_resnet18"#"DeepBrain"
    model_name_load="DeepBrain_Global_norm_2_suite_suite_final_model.pth"#"checkpoint_24.pth.tar"

    #checkpoint_24.pth.tar



    #add fine tuning parameters cf model.FT()
    parameters_training={"n_output":2#output of the hidden layer
                                ,"reset":"last_layer"#which layers to reinitialize : State of the art reset dense layers
                                ,"drop_out":True#useless currently
                                ,"freeze_type":"progressive"#type of freezing
                                ,"num_class":7,
                                "change_model":False}
    


    #add computing parameters
    dtype=torch.float32



    #PATH
    dic_path=get_dic_path()
    dic_path.update({"model_architecture":model_architecture,})
    parameters_training["param_save"] = dic_path

    ## Load hyperparameters


    #model selection
    num_class=parameters_training["num_class"]
    if model_architecture=="resnet18":
        model=S3ConvXFCResnet(27,num_class)
    elif model_architecture=="DeepBrain":
        model=DeepBrain()
    elif model_architecture=="RNN_resnet18":
        model=FEATURE_RNN_CF(n_classes=num_class)   
    w_load="./Records/trained_models/"+model_name_load
    D=torch.load(w_load)["state_dict"]
    model.load_state_dict(D)
        

    dataset, _ =get_torch_datasets(**param_data)
    loader = torch.utils.data.DataLoader(dataset,
                        batch_size=1,#batch_size=parameterization.get("batchsize", 3),
                        shuffle=False,
                        num_workers=2)
    path_data='../../../../../silenus/PROJECTS/pr-deepneuro/COMMON/DATA/INLANG/INNERSPEECH/FLATTEN_CONCATENATED/'
    for file in os.listdir(path_data):
        os.remove(path_data+file)
    model=model.Feature_extractor
    @torch.no_grad()
    def get_flatten(model,dataset_loader,dtype,device):
        path_data='../../../../../silenus/PROJECTS/pr-deepneuro/COMMON/DATA/INLANG/INNERSPEECH/FLATTEN_CONCATENATED/'
        t0 = time.time()
        y_pred=[]
        y_true=[]
        model.eval()
        model.to(device)

        #unshuffle
        file_list=dataset.file_list
        label_list=dataset.label_list
        L=list(zip(file_list,label_list))
        L.sort()
        file_list, label_list = [[i for i,j in L],[j for i,j in L]]
        dataset.file_list=file_list
        dataset.label_list=label_list


        for i in range(len(file_list[0])):
                if file_list[0][i:i+4]=="sub-":
                    string_to_remove=file_list[0][:i+4]
                    break
        for i,batch in enumerate(iter(dataset_loader)):
            
            ID=get_subject_ID([file_list[i]],len(string_to_remove))[0]
            x    = batch[0].to(dtype=dtype, device=device)
            label    = batch[1].type(torch.LongTensor)[0].item()
            features = model(x)[0].cpu().detach().numpy()
            np.save(path_data+ID+"_concatenated_flatten_innerspeech_global_norm_"+str(label)+".npy",features)
    get_flatten(model,loader,dtype,device)
