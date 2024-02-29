import pickle
from utilities.SPLIT import CV_generator
from utilities.Preprocessing_ML import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,balanced_accuracy_score, roc_auc_score
from sklearn.svm import SVC
import numpy as np
import time
def CrossValML(subject_ID_list, file_list, label_list, array_list, model, n_fold,
                path_save, preprocessing_dic, C_val="",kernel="",saving=False):
    """
        génére les modèles et leur prédiction respectives selon une méthode de validation croisée
        celle ci applique un preprocessing propre à chaque groupement de plis de donnée
    """
    path_models=path_save+"trained_models/"
    file_list=np.array(file_list)
    label_list=np.array(label_list)
    t0=time.time()

    t1=time.time()
    CV = CV_generator(file_list,subject_ID_list,n_fold=n_fold)
    list_model=[]
    print("time for convertion to nii :",t1-t0, " s")
    cpt=0
    f1_kfold_train=[]
    acc_kfold_train=[]
    f1_kfold_val=[]
    acc_kfold_val=[]
    P_list=[]
    for train_index_files, test_index_files in CV:
        #get data
        X_train=array_list[train_index_files]
        y_train=label_list[train_index_files]
        X_test=array_list[test_index_files]
        y_test=label_list[test_index_files]

        P_list+=[Preprocesser(**preprocessing_dic)]#init preprocesser
        P=P_list[-1]
        X_train=P.fit(X_train)#get attributes of preprocessing on training data and transform data
        X_test=P.transform(X_test)#transform test data

        #init model
        if model=="SVC":
            class_weight={i:(y_train==i).sum()/y_train.shape for i in np.unique(y_train)}#compute the weight of each class
            list_model+=[SVC(kernel=kernel,max_iter=-1,C=0.1**C_val,class_weight=class_weight)]
        elif model=="LDA":
            list_model+=[LDA(solver="svd")]
        decoder = list_model[-1]
        
        #train model
        decoder.fit(X_train,y_train)

        #get scores for the run
        y_pred_train=decoder.predict(X_train)
        y_pred_val=decoder.predict(X_test)
        if saving:
            save_ML_model(decoder,path_models,preprocessing_dic,model=model,kernel=kernel,fold=str(cpt),C=str(C))
            
        cpt+=1

        #append scores
        f1_kfold_train+=[f1_score(y_train,y_pred_train)]
        acc_kfold_train+=[balanced_accuracy_score(y_train,y_pred_train)]
        f1_kfold_val+=[f1_score(y_test,y_pred_val)]
        acc_kfold_val+=[balanced_accuracy_score(y_test,y_pred_val)]
        print("num fold :",cpt)

    print("@"*5)
    print("train")
    #print("cv_score",decoder.cv_scores_)
    print("f1_scores :",f1_kfold_train)
    print("acc_w_scores :",acc_kfold_train)
    
    return list_model, acc_kfold_train, f1_kfold_train, acc_kfold_val, f1_kfold_val, P_list


def save_ML_model(decoder,path_models,preprocessing_dic,model="svc",kernel="",fold="",C=""):
    
    if kernel:kernel="_"
    preprocessing_string=get_preprocessing_string(preprocessing_dic)
    s=path_models+f'{model}_{kernel}'+preprocessing_string+f'_fold_{fold}.pk'
    if C:
        s+="_C="+C
    with open(s+'.pk', 'wb') as pickle_file:
            pickle.dump(decoder,pickle_file)