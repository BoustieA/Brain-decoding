# -*- coding: utf-8 -*-


#Utilities
import os, fnmatch
import time
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,balanced_accuracy_score, roc_auc_score
#Model and training
import nibabel as nib


import time
from utilities.Preprocessing_ML import Preprocesser, average_prediction, get_preprocessing_string
from utilities.Train_func_ML import *
from utilities.Dataset import get_torch_datasets
from utilities.PATH import  get_dic_path
from utilities.SPLIT import get_subject_ID, get_train_test_index_from_ID, get_string_to_remove,CV_generator

if __name__=="__main__":
    print("###"*10)
    print("START SCRIPT")
    print("###"*10)
    t0 = time.time()
    
    print("#"*10)
    print()
    #PARAMETERS for single run

    #dic of preprocessing attributes/ cf Preprocessing_ML.Preprocesser __init__
    preprocessing_dic={"normalisation":"global",
                    "detrend":False,
                    "average_time":False,
                    "thresh_PCA":2}

    decoder_name="LDA"#"LDA" or "SVC", name of the decoder to use
    n_fold=10#nombre de plis pour la validation croisée
    groups="phono"
    path_save="./Records/LinearClassifier/"+decoder_name+"/"+groups+"/"
    

    #data choice
    param_data={"DATA":"PHONO_CONTROL_SMOOTHED",
                "split":"train_val",
                "env":"silenus",
                "crop":False}


    


    #train dataset : get file list and label list, the data is loaded as a whole onto CPU# again, get_torch_dataset is only used to retrieve the files
    train_set,val_set = get_torch_datasets(**param_data)
    file_list=train_set.file_list+val_set.file_list
    label_list=train_set.label_list+val_set.label_list
    subject_ID_list=get_subject_ID(file_list,len(get_string_to_remove(file_list, prefix="warsub-")),allow_control=False)#get subject ID for the split

    array_list=np.array([np.load(seq) for seq in file_list]).astype(np.float32)
    
    #test dataset
    param_data={"DATA":"PHONO_CONTROL_SMOOTHED",
        "split":"test",
        "env":"silenus",
        "len_seq":15,
        "processing":"raw",
        "normalization":None,
        "crop":False}

    #get test data, same principle as train_data
    _,test_set = get_torch_datasets(**param_data)
    file_list_test=test_set.file_list
    label_list_test=test_set.label_list
    array_list_test=np.array([np.load(seq)  for seq in file_list_test]).astype(np.float32)


    
    preprocessing_string=get_preprocessing_string(preprocessing_dic)

    #Training


    for kernel in ["linear"]:#,"sigmoid","rbf","poly"]:#loop onto kernel, unused if decoder="LDA"
        for C_val in [""]:#range(-10,-9,1):#loop onto C values, unused for "LDA decoder"
        #pour utiliser le decoder LDA,ne laisser qu'une itération de ces boucles
        
            print("n_sample :",len(train_set))
            #get the crossvalidation score estimate from training
            list_decoder, acc_train, f1_train, acc_val, f1_val, P_list =CrossValML(subject_ID_list, file_list, label_list, array_list, decoder_name, n_fold,
            path_save, preprocessing_dic, C_val=C_val,kernel=kernel,saving=False)
            
            
            print("end crossval")

            #compute average prediction on test set
            
            print("acc val")
            print(np.array(acc_val))
            y_pred_test=average_prediction(np.array(acc_val),array_list_test,list_decoder,P_list)
            print("@"*5)
            print("test")
            f1_test=f1_score(label_list_test,y_pred_test)
            acc_test=balanced_accuracy_score(label_list_test,y_pred_test)
            print("f1_score", f1_test)
            print("acc_w", acc_test)
            #create and write records in a log file 
            if decoder_name=="SVC":
                with open(path_save+"/"+"kernel_"+kernel+"_"+preprocessing_string+str(C_val)+"_"+".txt","w") as f:
                    f.write("C :\n")
                    f.write(f"{C_val}\n")
                    f.write("Train acc :\n")
                    f.write(str(acc_train)+"\n")
                    f.write("Val acc :\n")
                    f.write(str(acc_val)+"\n")
                    f.write("Test acc :\n")
                    f.write(str(acc_test)+"\n")
                    f.write("Train f1_score :\n")
                    f.write(str(f1_train)+"\n")
                    f.write("Val f1_score :\n")
                    f.write(str(f1_val)+"\n")
                    f.write("Test f1_score :\n")
                    f.write(str(f1_test)+"\n")
                    f.write("Test roc_auc :\n")
            elif decoder_name=="LDA":
                with open(path_save+"/"+preprocessing_string+".txt","w") as f:
                    
                    f.write("C :\n")
                    f.write(f"{C_val}\n")
                    f.write("Train acc :\n")
                    f.write(str(acc_train)+"\n")
                    f.write("Val acc :\n")
                    f.write(str(acc_val)+"\n")
                    f.write("Test acc :\n")
                    f.write(str(acc_test)+"\n")
                    f.write("Train f1_score :\n")
                    f.write(str(f1_train)+"\n")
                    f.write("Val f1_score :\n")
                    f.write(str(f1_val)+"\n")
                    f.write("Test f1_score :\n")
                    f.write(str(f1_test)+"\n")
                    f.write("Test roc_auc :\n")