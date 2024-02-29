# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:25:07 2023

@author: 33695
"""

"""
functions to retrieve files
"""
import numpy as np
import os, fnmatch
import ast



def get_dic_path():
    return{"path_models":"Records/trained_models",
            "path_curves":"Records/Evaluation/learning curves",
            "path_confusion_matrix":"Records/Evaluation/confusion_matrix",
            "path_transition_matrix":"Records/Evaluation/transition_matrix"
            }

def gather_list_label_file(read_path,labels,extension="nii.gz"):
    """
    

    Parameters
    ----------
    read_path : string
        path of the folder containing the files
    labels : list of string 
        conditions defined in file_name
    extension : string, optional
        extension of the files. 
        The default is "nii.gz".


    Returns
    -------
    file_list : list of string
        list of absolute path toward the sample of data
        
    label_list: list of string
        list of labels extracted from the file_name.

    """
    file_list=[]
    label_list=[]
    dic_label={label : i for i,label in enumerate(labels) }
    print("n_files to retrieve", len(os.listdir(read_path)))
    for path, folders, files in os.walk(read_path):#loop onto files entries
        for file in files:
            if fnmatch.fnmatch(file, '*'+extension):
                for label in labels:
                    if fnmatch.fnmatch(file, '*'+label+'*'+extension ):
                        label_list.append(dic_label[label])
                        file_list.append(os.path.join(read_path,file))
                        break#only one condition per file => break this inner loop
    
    L=list(zip(file_list,label_list))
    L.sort()
    file_list, label_list = [[i for i,j in L],[j for i,j in L]]

    return file_list, label_list



def get_files_label(DATA="HCP",env="silenus",extension="npy"):
    """
    fonction récupérant l'addresse des fichier selon les paramètres de la données
    a redéfinir si changement des chemins/dossier conteneur

    détermine les labels, l'extension et le chemin d'accès à la donnée

    retourne la liste des fichiers

    pour simplifier la fonction, garder la même structure entre bettik et silenus, mais pas toujours possible
    """
    path_data=f"/{env}/PROJECTS/pr-deepneuro/COMMON/DATA/"
    


    if DATA == "HCP":
        path_data += "HCP/RAW/"
        labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]
    elif DATA == "INNERSPEECH":
        if env=="bettik":
            path_data+="InLang/Innerspeech/tm2_float32"
        else:
            path_data+="INLANG/INNERSPEECH/RAW"
        labels=["_IMA_","_ISO_","_ISS_", "_SPP_","_VMW_"]
    elif DATA == "RECOVERY":
        if env=="bettik":
            path_data+="INLANG/Recovery/grouped_by15"
        else:
            path_data+="INLANG/Recovery/grouped_by15"
        labels=["-do_","-rep_","-rime_"]
    elif "SMOOTHED" in DATA:
        path_data+="INLANG/ReorgEpil/swar_smoothe/"
        if "PHONO_CONTROL" in DATA:
            path_data+="phono_control_grouped_by15"
            labels=["-phono_"]#["control-phono_"]
        elif "PHONO" in DATA:
            path_data+="phono_task_grouped_by15"
            labels=["-phono_"]
    elif DATA in ["PHONO", "PROSO", "SEM"]:
        path_data+=f"INLANG/ReorgEpil/war_non_smoothe/{DATA.lower()}_grouped_by15"
        labels=[f"-{DATA.lower()}_"]
        extension="npy"
    elif DATA in ["PHONO_CONTROL", "PROSO_CONTROL", "SEM_CONTROL"]:
        path_data+=f"INLANG/ReorgEpil/war_non_smoothe/{DATA.lower()}_grouped_by15"
        if "phono" in DATA.lower():
            labels=["CONTROL-PHONO".lower()]#[f"-{DATA.lower()}_"]
        elif "sem" in DATA.lower():
            labels=["control-sem".lower()]
    elif DATA == "INNERSPEECH_CONCATENATED":
        if env=="bettik":
            path_data+="InLang/Innerspeech/Innerspeech grouped_by27"
        else:
            path_data+="INLANG/INNERSPEECH/CONCATENATED"
        labels=["_IMA_","_ISO_","_ISS_", "_SPP_","_VMW_"]
    elif DATA == "INLANG":
        path_data+="../Data/INLANG/DATA_RAW"
        labels = ["gene","rap"]
    elif DATA=="AGE_SEMVIE":
        path_data+="INLANG/Age_groups/SEMVIE/grouped_by15_age_group"
        labels=["_young_","_old_"]
    elif DATA=="AGE_RECOVERY":
        path_data+="INLANG/Age_groups/Recovery/grouped_by15_age_group"
        labels=["_young_","_old_"]
    elif DATA=="AGE_NEUROMOD":
        path_data+="INLANG/Age_groups/NeuroMod/grouped_by15_age_group"
        labels=["_young_","_old_"]
    print(f"Gather data :{DATA} /hard_drive : {env} /format : {extension}")
    print(f"at :{path_data}")
    return gather_list_label_file(path_data,labels,extension)
