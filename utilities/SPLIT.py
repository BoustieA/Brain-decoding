# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:25:07 2023

@author: 33695
"""
"""
function to split the DATA onto subject ID

this functions use the facts that a sample files has the following string <path><subject_ID>*<condition>*<extension>
"""
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def get_subject_ID(file_list,len_string_to_remove,allow_control=True):
    """
    string_to_remove: 
    give the len of the path to remove before candidat_ID in file_name
    allow_control:
    when doing a control vs task classification, the subject is in both set, this option prevent duplicate


    return subject ID
    """
    subject_ID_list=[]
    for i,file_name in enumerate(file_list):
        
        if allow_control or "control" not in file_name:#otherwise duplicate ID +not same folder  => different string length
            f=file_name[len_string_to_remove:]
            if subject_ID_list==[] or subject_ID_list[-1] not in file_name:#check if subject is not already counted, otherwise redundancy may occur
                ID=""
                for s in f:
                    if s.isnumeric():
                        ID+=s
                    else:
                        break
                string=file_name[len_string_to_remove-1] + ID + s#to prevent partial ID such as _01_ to be identified in _101_
                
                subject_ID_list+=[string]
    #debug to remove
    return np.unique(subject_ID_list)

def get_train_test_index_from_ID(file_list, train_subjects, test_subjects):
    """
    get train test index according to subject ID. Only usefull when fusing multiple projects for cross validation purpose
    """
    train_index=[]
    test_index=[]
    for i,f in enumerate(file_list):
        for subject_id in train_subjects:
            if subject_id in f:
                train_index+=[i]
                break
        for subject_id in test_subjects:
            if subject_id in f:
                test_index+=[i]
                break
    return train_index, test_index


def split_train_test_from_ID(file_list,label_list,subject_ID_list):
    """
    from path_files_list,label_list and subject list
    return train/test files and labels
    """
    train_subject, test_subject = train_test_split(subject_ID_list, train_size=0.85 ,random_state=40)#split subjects ID into 2 set train and test
    print("n_subject_train :",len(train_subject))
    print(train_subject)#debug to confirm the separation of subjects
    print("n_subject_test :",len(test_subject))
    print(test_subject)#debug to confirm the separation of subjects
    train_index,test_index = get_train_test_index_from_ID(file_list, train_subject, test_subject)#get finals lists
    
    #select files and labels with index
    train_files=np.array(file_list)[train_index]
    train_labels=np.array(label_list)[train_index]
    test_files=np.array(file_list)[test_index]
    test_labels=np.array(label_list)[test_index]

    return train_files, train_labels, test_files, test_labels

def get_string_to_remove(file_list, prefix="sub-"):
    for i in range(len(file_list[0])):
            if file_list[0][i:i+len(prefix)]==prefix:#iterate onto the characters of the first file
                string_to_remove=file_list[0][:i+len(prefix)]#if the substring is "-sub" then stop the iteration, and keep the prefix of subject_id
                break
    return string_to_remove

def split_train_test_INLANG(file_list, label_list,split):
    #compute the string before subject ID which is supposed to be the same for all files
    string_to_remove=get_string_to_remove(file_list)


    subject_ID_list = get_subject_ID(file_list,len(string_to_remove))
    subject_ID_list.sort()
    test_portion=0.2#portion of subjects to keep for testing set

    if split=="train_val":
        ID_to_keep=subject_ID_list[:int(len(subject_ID_list)*(1-test_portion))]#keep only a portion of subject
        train_files, train_labels, test_files, test_labels = split_train_test_from_ID(file_list,label_list,ID_to_keep)#split them into train and validation data
        
        print("n_sample_train :",len(train_labels))
        print("n_sample_test :",len(test_labels))

        train_files,train_labels=shuffle(train_files,train_labels,random_state=42)
        return train_files, train_labels, test_files, test_labels
    elif split=="test":
        ID_to_keep=subject_ID_list[int(len(subject_ID_list)*(1-test_portion)):]#keep only a portion of subject

        #/!\ the following is a workaround to avoid the redefinition of sepcific functions, it only purpose is to select the files
        #according to the subjects and return the test set
        train_files, train_labels, test_files, test_labels = split_train_test_from_ID(file_list,label_list,ID_to_keep)#reuse splitting functions to select only testing subjects

        train_files = list(train_files)+list(test_files)#regroup files
        train_labels = list(train_labels)+list(test_labels)#regroup labels
        
        print("n_sample_test :",len(train_labels))
        print("test=train for finale evaluation")
        #return twice the files to allow the usage of the same function everywhere
        #virtually as training, and testing sample, in order to follow Wang Evaluation logic with dataset creation
        return train_files, train_labels, train_files, train_labels
    elif split=="all":
        return file_list, label_list, file_list, label_list
    elif split=="debug":
        return file_list[:10], label_list[:10],file_list[10:20], label_list[10:20]


    
def split_train_test_HCP(file_list, label_list, split="all"):
    if split=="debug":#keep only some files to debug pipelines
        return file_list[:200],label_list[:200], file_list[200:264], label_list[200:264]
    elif split == "train_val":#split files for training
        train_l = file_list[:20465]
        train_lab = label_list[:20465]
        val_l = file_list[20465:23350]
        val_lab = label_list[20465:23350]
        train_l,train_lab=shuffle(train_l,train_lab,random_state=42)#initial shuffle
        return train_l, train_lab, val_l,val_lab
    elif split=="test":#set for evaluation
        test_l = file_list[23350:29197]
        test_lab = label_list[23350:29197]
        #return twice the files to allow the usage of the same function everywhere
        #virtually as training, and testing sample, in order to follow Wang Evaluation logic with dataset creation
        return test_l, test_lab, test_l, test_lab
    elif split=="all":
        return file_list, label_list, [], []
    elif split=="sample":
        print("n_files ",len(file_list))
        len_string_to_remove=len("/silenus/PROJECTS/pr-deepneuro/COMMON/DATA/HCP/RAW/")#TODO
        subject_ID_list = get_subject_ID(file_list,len_string_to_remove)
        sample=0.3
        subject_ID_list=subject_ID_list[:int(len(subject_ID_list)*sample)]

        train_files, train_labels, test_files, test_labels = split_train_test_from_ID(file_list,label_list,subject_ID_list)
        print("n_files_train :",len(train_labels))
        print("n_files_val :",len(test_labels))
        """
        print("n_sample_train :",len(train_labels))
        print("n_subject_val :",len(test_subject))
        print("n_sample_val :",len(test_labels))
        """
        train_files,train_labels=shuffle(train_files,train_labels,random_state=42)#initial shuffle
        return train_files, train_labels, test_files, test_labels



def CV_generator(file_list,subject_ID_list,n_fold=2):

    """
        
    génère les index des différents plis pour une validation croisée selon les sujets/patients
    

        Parameters
        ----------
        file_list : liste des noms de fichier 
        subject_ID_list : liste des noms des sujets

        Returns les index de train et de test pour chaque itération
        -------

    """
    skf=iter(KFold(n_splits=n_fold,random_state=41,shuffle=True).split(subject_ID_list))#créer un itérateur de selection de plis

    for i in range(n_fold):#pour chaque combinaison de plis de sujets, récupère les indexs de fichiers correspondant
        train_index, test_index = next(skf)#index des sujets

        train_index=list(train_index)
        test_index=list(test_index)
        train_subjects = np.array(subject_ID_list)[train_index]#combinaison de sujets du run
        test_subjects = np.array(subject_ID_list)[test_index]#combinaison de sujets du run

        #ligne de debug, affiche la répartition train/test
        print("subject for run ", i)
        print("train subjects :\n", train_subjects)
        print("test subjects :\n",test_subjects)
        #récupéère les noms de fichier correspondant
        train_index_files,test_index_files = get_train_test_index_from_ID(file_list, train_subjects, test_subjects)
        yield train_index_files, test_index_files