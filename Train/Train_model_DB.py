# -*- coding: utf-8 -*-


#Utilities
import os, fnmatch
import time
import torch

#import dill as pickle
# and training
from utilities.Savemodel import save_model
from utilities.PATH import get_dic_path
from utilities.Dataset import get_torch_datasets
from utilities.Train_func_NN import Training
import matplotlib.pyplot as plt

#Models
from Models.DeepBrain_wang import DeepBrain
from Models.resnet18 import S3ConvXFCResnet
from Models.RNN_resnet18_depthwise import FEATURE_RNN_CF
#from torch.utils.data import random_split
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle

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

    model_name_save="Debugload"
    SAVING=True#wether to save or not the model

    #model_type
    model_architecture="resnet18"#"RNN_resnet18"#"DeepBrain"

    #data choice
    param_data={"DATA":"PHONO_CONTROL_SMOOTHED",
                "processing":"oversample",
                "env":"silenus",
                "len_seq":14,
                "split":"debug",
                "normalization":"time"}

    #hyperparameters
    """
    parameters_training={"lr":0.001
                        ,"momentum":0.9
                        ,"batch_size":4
                        ,"num_epochs":2
                        ,"scheduler":"onecycle"
                        ,"max_lr":0.01
                        ,"optimizer":"ADAM"
                        ,"step_size":15
                        ,"gamma":0.1
                        ,"loss_weights":None
                        ,"num_class":7}
    """

    base_lr=0.01
    parameters_training={
                    #param calcul
                    "mixed_precision":True#pour entrainer plus vite et mettre des plus gros batch
                    ,"num_worker":2#nombre de worker, parallélise le chargement de la donnée à l'entrainement
                                   #4 max sur le noeud de dev, pour charger la donnée plus vite, 8 sur les autres noeuds. Mais dépend de la complexité des preprocessing
                    ,"DDP":False#pour la parallélisation de l'entrainement, pas très utile et ne fonctionne pas actuellement (normalement les fonctions sont clean pour les boucles d'entrainements, pas forcément pour les optimiser, loss ....)
                    
                    #param entrainement
                    ,"lr":base_lr#0.0000001
                    ,"batch_size":8#taille de batch <16 pour un petit GPU c'est pas catastrophique pour notre jeu de donnée, au contraire
                    ,"num_epochs":3
                    ,"optimizer":"Adam"# Adam ou SGD, SGD plus dur a tune
                    ,"momentum":0.9#utile pour SGD, sinon inutilisé

                    #param scheduler
                    ,"scheduler":"plateau"#"onecycle"#/"step"/"plateau"
                        #onecycle, learning rate incrémental jusqu'à un max lr, puis décroissant
                            ,'max_lr':base_lr*10#[base_lr,base_lr,base_lr]#une liste avec les learning rate différentiel, un entier sinon
                        #"step"
                            ,"step_size":10 #nombre d'epoch avant decay du learning rate
                        #plateau, le learning rate est diminué tout les "step_size" epoch où la loss n'a pas diminuée
                            ,"gamma":0 #de 0 à 1, valeur de diminution du lr avec lr=lr*gamma
                            ,"step_size":10#nombre d'epoch de patience
                        #

                    #regularisation
                    ,"loss_weights":True#pour la donnée imbalanced, calculé par la classe dataset
                    ,"label_smoothing":0.1 # entre 0 et 1, ça a été inventé pour géré les mauvaise labellisation, probablement inutile ici
                    ,"L2_reg":0.001#L2 reg par rapports aux poids d'origine, pour le transfert learning
                    ,"weights_decay":False#L2 reg pour l'entrainement initial d'un modèle
                    ,"num_class":2}
    #FT param
    parameters_training["LOAD"]=False#if continuing a training session from a previous recorded state
    model_name_load="./Records/trained_models/"+"Debug_best_model.pth"
    #"DeepBrain_Global_norm_2_suite_suite_final_model.pth"#path toward weights to load if LOAD is True
    parameters_training["model_name_load"]=model_name_load
    #checkpoint_24.pth.tar #wang model DeepBrain


    #add computing parameters
    parameters_training.update({"DDP":False# enable multiGPU #not working yet
                            , "num_workers":2#num of workers in data loader int>1 allow to load next batch on gpu while computing grad of current batch
                            ,"mixed_precision":True})
    dtype=torch.float32





    #PATH
    dic_path=get_dic_path()
    dic_path.update({"model_architecture":model_architecture,
                    "model_name":model_name_save,})
    parameters_training["param_save"] = dic_path
    parameters_training["param_save"]["save"]=SAVING






    #dataset
    train_set,val_set = get_torch_datasets(**param_data)

    #get_n_sample, required for onecycle scheduler
    parameters_training["n_sample"]=len(train_set)

    #get_weights for regularized unbalanced data
    if parameters_training["loss_weights"]:
        parameters_training["loss_weights"]=torch.FloatTensor(train_set.get_weights_labels()).to(device)
        print("adding loss weights :")
        print(parameters_training["loss_weights"])
    else:
        parameters_training["loss_weights"]=None

    #model selection, create architecture
    num_class=parameters_training["num_class"]
    if model_architecture=="resnet18":
        model=S3ConvXFCResnet(27,num_class)
    elif model_architecture=="DeepBrain":
        model=DeepBrain()
    elif model_architecture=="RNN_resnet18":
        model=FEATURE_RNN_CF(n_classes=num_class)   
        
    
    t2 = time.time()


    dic_save=dict(parameters_training["param_save"])
    parameters_training["param_save"]=dic_save

    #display parameters in the logs
    print("Parameters :")
    for i in parameters_training:
        if "state_dic" not in i:
            print(i," : ",parameters_training[i])
    print("data :",param_data)
    print("n_sample train :",len(train_set))
    print("n_sample val:",len(val_set))


    ########### start the training ############


    #Training



    
    Train=Training(model,parameters_training,dtype)#setup Trainer object, 


    Train.fit(model,train_set,val_set,save_best_model=True)#fit the model

    
    history = Train.history      



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




    ##Save model final_model
    if parameters_training["param_save"]["save"]:
        dic_save=dict(parameters_training["param_save"])
        dic_save["model_name"]=dic_save["model_name"]+"_final_model"
        save_model(range(1,Train.num_epochs+1), model, Train.optimizer, Train.criterion, Train.scheduler,dic_save)

    print(history)










    
