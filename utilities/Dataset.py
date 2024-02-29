 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:51:51 2022

@author: neurodeep
"""
import os, fnmatch

import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

from torch import nn
import torch
from utilities.PATH import *
from utilities.SPLIT import *

class BrainDataset(Dataset):
    def __init__(self, file_list, label_list, is_train=True,
    extension="nii.gz",input_shape = 27,normalization="global",
    crop=False, prevent_negative=False):
        """
        classe pytorch pour le preprocessing de la donnée en entrée du modèle
        charge la donnée à partir de son chemin puis applique les différent preprocessing

        """
        self.is_train = is_train
        self.input_shape = input_shape #taille de séquence à utiliser
        self.file_list = file_list#list des chemins des échantillons
        self.label_list = label_list#liste des labels respectifs
        self.extension=extension#extension nii.gz ou npy
        self.normalization=normalization#type de normalisation à effectuer
        self.crop_=crop#redimensionnement (booléen)
        self.prevent_negative=prevent_negative#dans le dataset Inlang peu de valeurs négatives existe, mais sont mise à zéro avec cette option par défaut
    
    def __getitem__(self, index):
        """
        fonction de récupération de la donnée,
        quand la classe est mise sous la forme d'un itérateur : iter(BrainDataset(**kwargs))
        retourne un échantillon dans l'ordre de la liste
        en général l'appel est transparent au travers d'un dataloader pour la gestion de chargement parallèle, batch ...
        """
        data, target = self.load_data(index) #charge le fichier
        data = self.crop(data)#coupe les bords si nécessaire
        data = self.extract_seq(data)#extrait la séquence
        data = self.normalize_data(data)
        data = data.transpose(3, 0, 1, 2)
        return data, target
    
    def __len__(self):
        return len(self.label_list)
    
    def load_data(self,index):
        if self.extension=="nii.gz":
            data = nib.load(self.file_list[index]).get_fdata()
        elif self.extension=="npy":
            data = np.load(self.file_list[index])
            #data = np.load(self.file_list[index], mmap_mode="r") selon les usages (slicing dans un gros tenseur) peut être plus rapide pour charger la donnée
        target = self.label_list[index]
        return data, target

    def extract_seq(self,data):
        if self.input_shape==None:
            #pas de crop
            return data
        if self.is_train:
            #pour entrainement séquence contigue aléatoire,cf Wang et al. https://doi.org/10.1002/hbm.24891
            len_seq=data.shape[-1]
            start_seq=np.random.randint(0,len_seq+1-self.input_shape)
            return data[:,:,:, start_seq:start_seq+self.input_shape]
        else:
            #pour l'évaluation extraction de frames contigues
            return data[:,:,:, :self.input_shape]

    def crop(self,data):
        if self.crop_:
            return data[8:-8, 8:-8, :-10, :]
        else:
            return data

    def normalize_data(self, data):
        """
        normalize la donnée :
        global: normalise la donnée par rapport au paramètre "maximum" définit sur l'intégralité des voxels de la séquence de l'échantillon
        time: normalise chaque séquence de voxels séparément
        max: normalise la donnée relativement à un maximum de tout les échantillons à définir


        la donnée est supposé appartenir à l'interval [0,N] pour l'entiereté des images IRMf, et [1,N] pour le volum cérébral.
        de sorte qu'il n'y a pas de plage de donnée non utilisé.

        Pour la donnée provenant de Inlang, des valeurs négatives sont présentes en faible quantité.
        Le paramètre prevent_negative les transforme en 0

        """
        if self.normalization=="global":
            if self.prevent_negative:
                data=(data>0)*data
            
                #data=(data-data.min()*mask)
            else:
                data = data - data.min()
            data =data/data.max()
        elif self.normalization=="time":
            data = data - data.min(axis=3)[:, :, :, np.newaxis]
            data= data / data.max(axis=3)[:, :, :, np.newaxis]
            data[~ np.isfinite(data)] = 0
        elif self.normalization=="max":
            if self.prevent_negative:
                data=(data>0)*data
                #data=(data-data.min()*mask)
            else:
                data = data - data.min()
            data=data/12240
        elif self.normalization=="mean":
            if self.prevent_negative:
                data=(data>0)*data
            filled=np.count_nonzero(data)#count values outside of 0
            mean=data.sum()/filled
            data=data/mean
        return data#.transpose(3, 0, 1, 2)
    
    def get_weights_labels(self):
        """
        Fournit le paramètre d'importance de chaque classe pour atténuer les déséquilibres dans l'entrainement.
        Calcul tout d'abord la proportion d'échantillon par classe, 
        puis détermine le poids d'importance en prenant l'inverse de ce ratio
        """
        weights=[]
        for i in np.unique(self.label_list):#calcul des ratio par classe
            weights+=[(np.array(self.label_list)==i).sum()]
        print("balance data :",weights)
        weights=1/np.array(weights)#poids de régularisation du déséquilibre
        return weights/weights.sum()#normalisation de ces poids
















class TransformersDataset(BrainDataset):
    """
    classe pour formater la donnée selon le format transformers
    ici, la donnée est paddé selon les paramètres nécessaire pour le modèle de model Attentionnel de Wang
    taille de séquence du modèle : 15
    """
    def __init__(self, file_list, label_list, is_train=True,
    extension="nii.gz",input_shape = 27,normalization="global",
    crop=False, prevent_negative=False):
        super().__init__(file_list, label_list, is_train,
        extension,input_shape,normalization,
        crop,prevent_negative)
    def __getitem__(self, index):
        data, target = self.load_data(index) 
        data = self.crop(data)
        data = self.extract_seq(data)
        data = self.normalize_data(data)
        data = data.transpose(3, 0, 1, 2)

        data_ = np.zeros((15,80, 96, 88))#padding
        data_[:,3:-2,2:-1,:-7]=data

        data_=data_[None,:,:,:,:]#expect 5D tensor : channel, time sequence, spatial
        return data_, target
    
    
    
    def extract_seq(self,data):
        #input sequence for attention model is 15
        #padding is used to match such a size when files are 14 frames long
        len_seq=data.shape[-1]
        if self.input_shape==None:
            return data
        if self.is_train:
            if len_seq==14:#padding randomly for training
                return np.concatenate([data,data[:,:,:,-1:]],axis=-1)
                if np.random.randint(0,2)==0:#padding end or start randomly
                    return np.concatenate([data[:,:,:,:1],data],axis=-1)#duplicate first frame
                else:
                    return np.concatenate([data,data[:,:,:,-1:]],axis=-1)#duplicate last frame
            
            start_seq=np.random.randint(0,len_seq+1-self.input_shape)
            return data[:,:,:, start_seq:start_seq+self.input_shape]
        else:
            if len_seq==14:# padding for testing
                return np.concatenate([data,data[:,:,:,-1:]],axis=-1)
            return data[:,:,:, :self.input_shape]
    




class BrainDatasetOverSample(BrainDataset):
    """
    Oversample de la séquence temporelle par interpolation simple
    la taille de la séquence du fichier doit être supérieur ou égal à 14
    la taille de la séquence interpolé est de 27
    """
    def __init__(self, file_list, label_list, is_train=True,
        extension="nii.gz",input_shape = 14,normalization="global",
        crop=False,prevent_negative=True):
        super().__init__(file_list, label_list, is_train,
        extension,input_shape,normalization,
        crop,prevent_negative)


    def __getitem__(self, index):
        data, target = self.load_data(index)    
        data = self.crop(data)
        data = self.extract_seq(data).transpose(3, 0, 1, 2)
        data = self.normalize_data(data)
        data = self.oversample(data)#ajout de l'interpolation
        return data, target
    
    def oversample(self,data):
        C=np.zeros((27,75,93,81))
        for i in range(13):
            C[0+i*2,:,:,:]=data[i,:,:,:]
            C[1+i*2,:,:,:]=data[i,:,:,:]/2+data[i+1,:,:,:]/2
        C[26,:,:,:]=data[13,:,:,:]
        return C
    






class FlattenData(Dataset):
    def __init__(self, file_list, label_list, is_train=True):
        self.is_train = is_train
        self.file_list = file_list
        self.label_list = label_list

    def __getitem__(self, index):
        #img = self.load(torch.load(self.file_list[index]))
        img = self.file_list[index].ravel()
        target = self.label_list[index]
        return img, target
        
    def __len__(self):
        return len(self.label_list)

    def get_weights_labels(self):
        weights=[]
        for i in torch.unique(self.label_list):
            weights+=[(self.label_list==i).sum()]
        weights=torch.Tensor(weights)
        print("balance data :",weights)
        weights=1/weights
        return weights/torch.sum(weights)




def get_ALL_INLANG(processing,split,env,extension):
    """
    charge toute la base InLang puis incrémente les labels

    labels =["_IMA_","_ISO_","_ISS_", "_SPP_","_VMW_","-do_","-rep_","-rime_"]#,"-phono_", "-proso_", "-sem_"]
    #cluster : 0/0/0/1/2/3/3/4/4/4/1
    {0:0,1:0,2:0
    ,3:1,
    4:2,
    5:3,6:3,
    7:4,8:4,9:4,
    10:1}
    """
    train_l,train_lab,val_l,val_lab=[],[],[],[]
    last_label=0
    for DATA in ["INNERSPEECH","RECOVERY","PROSO","PHONO", "SEM"]:
        #retrieve files
        file_list, label_list  = get_files_label(DATA=DATA,env=env,extension=extension)
        #split
        train_l1,train_lab1,val_l1,val_lab1  = split_train_test_INLANG(file_list, label_list,split=split)#the same random state is used for spliting control and test
        #update label to concatenate them
        train_lab1=[int(label + last_label) for label in train_lab1]
        val_lab1=[int(label + last_label) for label in val_lab1]
        last_label=np.max(train_lab1)+1#update last label +1 to avoid overlap

        #update list
        train_l+=list(train_l1)
        val_l+=list(val_l1)
        train_lab+=train_lab1
        val_lab+=val_lab1
    
    return train_l,train_lab,val_l,val_lab


def get_CONTROL_VS_TASK_INLANG(data,processing,split,env,smoothed,extension):
    #data is under format condition_control_(smoothed)
    condition=data.split("_")[0]
    train_l,train_lab,val_l,val_lab=[],[],[],[]
    last_label=0
    for DATA in [condition,condition+"_CONTROL"]:
        #retrieve files
        DATA_=DATA
        if smoothed:
            DATA_="SMOOTHED_"+DATA
    
        file_list, label_list  = get_files_label(DATA=DATA_,env=env,extension=extension)
        #split
        train_l1,train_lab1,val_l1,val_lab1  = split_train_test_INLANG(file_list, label_list,split=split)#the same random state is used for spliting control and test
        #update label to concatenate them
        train_lab1=[int(label + last_label) for label in train_lab1]
        val_lab1=[int(label + last_label) for label in val_lab1]
        last_label=np.max(train_lab1)+1#update last label +1 to avoid overlap
        #update list
        train_l+=list(train_l1)
        val_l+=list(val_l1)
        train_lab+=train_lab1
        val_lab+=val_lab1
    
    return train_l,train_lab,val_l,val_lab



def get_AGE_INLANG(processing,split,env,extension):
    train_l,train_lab,val_l,val_lab=[],[],[],[]
    last_label=0
    for DATA in ["AGE_RECOVERY","AGE_NEUROMOD"]:#,"AGE_SEMVIE"
        #retrieve files
        file_list, label_list  = get_files_label(DATA=DATA,env=env,extension=extension)
        #split
        train_l1,train_lab1,val_l1,val_lab1  = split_train_test_INLANG(file_list, label_list,split=split)
        #update label to concatenate them
        train_lab1=[int(label + last_label) for label in train_lab1]
        val_lab1=[int(label + last_label) for label in val_lab1]
        last_label=np.max(train_lab1)+1#update last label +1 to avoid overlap

        #update list
        train_l+=list(train_l1)
        val_l+=list(val_l1)
        train_lab+=train_lab1
        val_lab+=val_lab1
        last_label=0
    
    return train_l,train_lab,val_l,val_lab


def get_torch_datasets(DATA="HCP",processing="max",split="all",env="silenus"
                ,len_seq=27, normalization="global", crop=False):
    """
    env: source of data ("silenus"/"bettik")
    split : type of splitting : "all":all data
                                "test":test ssample
                                "train_val":train and validation samples
                                "debug":some sample to debug
    DATA: type of data 
        "HCP",
        "PHONO_CONTROL","SEM_CONTROL",
        "INLANG",INLANG_AGE


    len_seq: number of contiguous frame to retrieve
    processing : type of preprocessing needed : "max"/"raw"#usefull if the files are already preprocessed in storage#unused currently
    
        NB:
            -GBP same as raw
            -raw is .nii.gz on bettik vs cropped array on silenus

    example :

    param_data={"DATA":"PHONO_CONTROL_SMOOTHED",
            "processing":"oversample",#"raw",transformers
            "split":"train_val",#"test","debug"
            "env":"silenus",#bettik
            "len_seq":15,#15 for attention of wang, 14 for oversample, no constraint else
            "normalization":"global",#"mean","max","time" #cf braindataset normalisation function
            }#train_val_test #test
    train_set,val_set=get_torch_datasets(**param_data)
    """
    #paramètres relatifs à la données probablement à sortir
    extension, crop, prevent_negative,smoothed=get_param_data(DATA,env)
    #get_files

    #récupération de la liste des fichier du dataset correspondant

    #plusieurs dossiers
    if DATA=="INLANG":
        train_l,train_lab,val_l,val_lab  = get_ALL_INLANG(processing,split,env,extension)
    elif "CONTROL" in DATA:
        train_l,train_lab,val_l,val_lab  = get_CONTROL_VS_TASK_INLANG(DATA,processing,split,env,smoothed,extension)
    elif DATA=="INLANG_AGE":
        train_l,train_lab,val_l,val_lab  = get_AGE_INLANG(processing,split,env,extension)
    else:
        #un seul dossier
        file_list, label_list  = get_files_label(DATA=DATA,env=env,extension=extension)
        if DATA=="HCP":
            train_l,train_lab,val_l,val_lab  = split_train_test_HCP(file_list, label_list,split=split)
        elif "INNERSPEECH" in DATA or "RECOVERY" in DATA or "AGE" in DATA:
            train_l,train_lab,val_l,val_lab  = split_train_test_INLANG(file_list, label_list,split=split)    

    if processing.lower()=="transformers":
        train_set = TransformersDataset(train_l, train_lab, is_train=True,
                extension=extension,input_shape=len_seq,
                normalization=normalization, crop=crop,prevent_negative=prevent_negative)
        val_set = TransformersDataset(val_l, val_lab, is_train=False,
                extension=extension,input_shape=len_seq,
                normalization=normalization, crop=crop,prevent_negative=prevent_negative)

    elif processing.lower()=="oversample":
        train_set = BrainDatasetOverSample(train_l, train_lab, is_train=True,
                extension=extension,input_shape=len_seq,
                normalization=normalization, crop=crop,prevent_negative=prevent_negative)
        val_set = BrainDatasetOverSample(val_l, val_lab, is_train=False,
                extension=extension,input_shape=len_seq,
                normalization=normalization, crop=crop,prevent_negative=prevent_negative)
    elif processing.lower()=="oversample_create_flatten":#for flatten creation return dataset on whole data in validation mode
        train_set = BrainDatasetOverSample(train_l+val_l, train_lab+val_lab, is_train=False,
                extension=extension,input_shape=len_seq,
                normalization=normalization, crop=crop,prevent_negative=prevent_negative)
        val_set = train_set
    elif processing.lower()=="create_flatten":#for flatten creation return dataset on whole data in validation mode
        train_set = BrainDataset(train_l+val_l, train_lab+val_lab, is_train=False,
                extension=extension,input_shape=len_seq,
                normalization=normalization, crop=crop,prevent_negative=prevent_negative)
        val_set = train_set
    elif processing.lower()=="raw":
        train_set = BrainDataset(train_l, train_lab, is_train=True,
                extension=extension,input_shape=len_seq,
                normalization=normalization, crop=crop,prevent_negative=prevent_negative)
        val_set = BrainDataset(val_l,val_lab, is_train=False,
                extension=extension,input_shape=len_seq,
                normalization=normalization, crop=crop,prevent_negative=prevent_negative)

    
    else:
        train_set = BrainDataset(train_l, train_lab, is_train=True,
                extension=extension,input_shape=len_seq,
                normalization=normalization, crop=crop,prevent_negative=prevent_negative)
        val_set = BrainDataset(val_l,val_lab, is_train=False,
                extension=extension,input_shape=len_seq,
                normalization=normalization, crop=crop,prevent_negative=prevent_negative)

    return train_set, val_set

#processing :oversample/transformers
#DATA INLANG/INNERSPEECH

#extension
#crop
#normalisation
#prevent negative

#param dataset
#

#at hand

#processing
#normalisation
#DATA

def get_param_data(DATA="HCP",env="silenus"):
    """
    retourne les paramètres propres aux fichiers selon les environnements de stockage prédéfinis
    extension, crop si nécessaire, prevent negative, lissage
    """

    is_INLANG="INNERSPEECH" in DATA or "RECOVERY" in DATA or "RegorgEPIL" in DATA or "INLANG" in DATA or "AGE" in DATA
    if is_INLANG:
        prevent_negative=True
    else:
        prevent_negative=False
    
    if env.lower()=="bettik":#TODO unclean : better do same folder name = same extension
        if processing.lower() in ["raw","oversample","create_flatten"]:
            if DATA=="HCP":
                crop=True
                extension="nii.gz"
            elif is_INLANG:
                extension="npy"
                crop=False
    else:
        extension="npy"
        crop=False
    #get_files
    smoothed=False
    if "smoothed" in DATA.lower():      
        smoothed=True
    return extension, crop, prevent_negative,smoothed