# -*- coding: utf-8 -*-


#Utilities
import os, fnmatch
import time
import torch
import numpy as np
import dill as pickle
import torch.nn as nn
from utilities.Savemodel import save_model
from utilities.Dataset import get_torch_datasets, BrainDatasetOverSample
from utilities.PATH import  get_dic_path
from utilities.SPLIT import get_subject_ID,CV_generator,get_string_to_remove
from utilities.Train_func_NN import Training
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,balanced_accuracy_score
import seaborn as sns

#Model and training
from Models.Attention_net_wang import Attention_network
from Models.DeepBrain import DeepBrain
from Models.resnet18 import S3ConvXFCResnet
#from Models.RNN_resnet18_depthwise_wide import FEATURE_RNN_CF
from Models.CNN_LSTM import FEATURE_RNN_CF2 as FEATURE_RNN_CF
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


def plot_history(Train,parameters_training):
    """
    plot learning curves
    """
    history = Train.history      
    model_name_save=parameters_training["param_save"]["model_name"]


    ## Plots


    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)


    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    path_image=os.path.join(parameters_training["param_save"]["path_curves"],f'{model_name_save}_accuracy.png')
    plt.savefig(path_image)
    plt.close()



    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    path_image=os.path.join(parameters_training["param_save"]["path_curves"],f'{model_name_save}_loss.png')
    plt.savefig(path_image)
    plt.close()

    
def init_model():
    """
    initialisation of the model
    wang, Resnet or Transformers,
    Preprocessing are defined accordingly
    """
    #model="attention_network"
    model="resnet"
    #model="wang"
    #get pretrained model
    print("Model for Feature Extraction :")
    print(model)
    classifier_param={}
    if model.lower()=="attention_network":
        print("Attention network")
        model=Attention_network()
        w_load="./Records/trained_models/Attention_net_weights.pth.tar"
        D=torch.load(w_load)["state_dict"]
        model.load_state_dict(D)
        param_data["normalization"]="mean"
        param_data["processing"]="transformers"
        param_data["len_seq"]=15
        classifier_param["input_size"]=128
    elif model.lower()=="resnet":
        print("Resnet")
        model=S3ConvXFCResnet(27,7)
        w_load="./Records/trained_models/DeepBrain_Global_norm_2_suite_suite_final_model.pth"
        D=torch.load(w_load)["state_dict"]
        model.load_state_dict(D)
        #model=model.Feature_extractor
        print("last layer")
        param_data["normalization"]="global"
        param_data["processing"]="oversample"
        param_data["len_seq"]=14
        classifier_param["input_size"]=512
    else:
        model=DeepBrain()
        w_load="./Records/trained_models/checkpoint_24.pth.tar"
        #D=torch.load(w_load)["state_dict"]
        model.load_model(w_load)#recrée l'architecture et charge le modèle pour avoir le même format que le resnet :Feature_extractor/classifier
        #model=model.Feature_extractor
        print("last layer")
        param_data["normalization"]="temporal"
        param_data["processing"]="oversample"
        param_data["len_seq"]=14
        classifier_param["input_size"]=64
    model.classifier=nn.Sequential(#nn.Dropout(0.5),#0.25#0.45
    nn.Linear(classifier_param["input_size"],2)
    )
    #dataset
    """
    #freezing f needed
    for i,m in enumerate(model.Feature_extractor[1].modules()):
        
        if i in [4,17,33,49]:
            #print(i)
            #print(m)
            if i<0:
                print("block frozen")
                for p in m.parameters():
                    p.requires_grad=False
    """
    #for i in model.Feature_extractor[0].parameters():
    #    i.requires_grad=False
    return model

def CrossVal(training_parameters, subject_ID_list, file_list, label_list,n_fold=10):
    """
    Entraine plusieurs modèles pour différentes combinaisons de plis de données
    """
    path_models="./Records/trained_models/"
    file_list=np.array(file_list)
    label_list=np.array(label_list)
    CV = CV_generator(file_list,subject_ID_list,n_fold=n_fold)#générateurs des indexes de  plis de données selon les sujets
    dic_param_save=dict(training_parameters["param_save"])
    loss_kfold=[]
    acc_kfold=[]
    f1_kfold=[]
    list_model=[]
    for i,indexes in enumerate(CV):#pour chaque combinaison de plis de données
        train_index,test_index=indexes
        torch.cuda.empty_cache()
        training_parameters["param_save"]["model_name"]=dic_param_save["model_name"]+"_"+str(i)
        
        #get datasets
        train_index=list(train_index)
        test_index=list(test_index)
        
        X_train=file_list[train_index]
        y_train=label_list[train_index]
        X_test=file_list[test_index]
        y_test=label_list[test_index]

        #create dataset with data sample, /!\ use the appropriate class
        train_set=BrainDatasetOverSample(X_train,y_train,extension="npy",crop=False,normalization="global"
                ,input_shape=14,is_train=True)
        val_set=BrainDatasetOverSample(X_test,y_test,extension="npy",crop=False,normalization="global"
        ,input_shape=14,is_train=False)

        _model=init_model()#init model  each iter

        #add some parameters for training
        training_parameters["n_sample"]=len(train_set)
        W=torch.Tensor(train_set.get_weights_labels()).to(device=device)
        training_parameters["loss_weights"]=W

        #instanciate trianing class
        Train=Training(_model, training_parameters, dtype)
        
        #fit the model
        _model=Train.fit(_model, train_set, val_set, save_best_model=True,verbose=False) 
        #plot the learning curves for this run
        plot_history(Train,training_parameters)

        #load best model according to val data from this run then evaluate
        D=torch.load(path_models+training_parameters["param_save"]["model_name"]+"_best_model.pth")["state_dict"]
        _model.load_state_dict(D)

        #evaluate
        f1,acc=Train.evaluate(_model,val_set,other_scores=True)
        acc_kfold+=[acc]
        f1_kfold+=[f1]
        
        #update a list of model name to print in the log at the end of the script
        list_model+=[training_parameters["param_save"]["model_name"]]

        #clean GPU/CPU for storage capacity
        del _model
        del D
        del Train
        collected = gc.collect()
        #loss_kfold+=[loss]
    #print in the logs some metrics
    print("f1_scores :",f1_kfold)
    print("mean :",np.mean(f1_kfold))
    print("std :",np.std(f1_kfold,ddof=1))

    print("acc_w_scores :",acc_kfold)
    print("mean :",np.mean(acc_kfold))
    print("std :",np.std(acc_kfold,ddof=1))
    return list_model, acc_kfold, f1_kfold



#/!\ définir la fonction init model et le dataset dans la fonction CrossVal selon le modèle et preprocessing a utiliser
#ici pour le finetuning une stratégie d'oversampling par interpolation est utilisé, un dataset spécifique est utilisé

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
    #PARAMETERS for single run
    
    

    n_outputs=5

    model_name_save="debug"
    SAVING=False#save the models
    #model_type

    #data choice
    param_data={"DATA":"PHONO_CONTROL_SMOOTHED",
            "split":"train_val",
            "env":"silenus",
            "crop":False}

    #hyperparameters
    base_lr=0.001/10
    parameters_training={"lr":base_lr#0.0000001
                        ,"momentum":0.9
                        ,"batch_size":8
                        ,"num_epochs":1
                        ,"optimizer":"Adam"
                        ,"scheduler":"plateau"
                        ,"gamma":0.99
                        ,"step_size":10
                        ,"loss_weights":True
                        ,"num_class":2
                        ,"mixed_precision":True
                        ,"num_worker":2
                        ,"DDP":False
                        ,'max_lr':base_lr*10#[base_lr/10,base_lr,base_lr*10]#0.0001
                        ,"label_smoothing":0}
    
    total_trial=200


    #add computing parameters
    dtype=torch.float32




    #Autonomous scripts

    #PATH
    dic_path=get_dic_path()
    dic_path.update({"model_name":model_name_save})
    parameters_training["param_save"] = dic_path
    parameters_training["param_save"]["save"]=SAVING



    ## Load hyperparameters




    
    t2 = time.time()
    
    #create and join the datasets for CV# only the files are used, since training and evaluating involve different preprocessing
    train_set,val_set = get_torch_datasets(**param_data)
    file_list=train_set.file_list+val_set.file_list
    label_list=train_set.label_list+val_set.label_list
    subject_ID_list=get_subject_ID(file_list,len(get_string_to_remove(file_list, prefix="warsub-")),allow_control=False)


    print("parameters :",parameters_training)
    #print("data :",param_data)
    print("n_sample :",len(train_set))

    #Training

    
    list_model, acc_kfold, f1_kfold =CrossVal(parameters_training, subject_ID_list, file_list, label_list)
    #print in the logs, the metrics and the name of the model saved
    print("end crossval")
    print("model saved")
    print(list_model)
    print("accuracy balanced")
    print(acc_kfold)
    print("f1_weightened")
    print(f1_kfold)









    
