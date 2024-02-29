# -*- coding: utf-8 -*-

"""
Created on Sat Feb  4 12:48:23 2023

@author: 33695
"""


import functools as fntls
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from Models.resnet18 import S3ConvXFCResnet
import numpy as np
import os

from sklearn.metrics import f1_score as f1_score_
from sklearn.metrics import f1_score,balanced_accuracy_score
from torch.utils.data.distributed import DistributedSampler

from utilities.Savemodel import SaveBestModel, save_model
"""
example of parameter training
parameters_training={
                    #param calcul
                    "mixed_precision":True#pour entrainer plus vite et mettre des plus gros batch
                    ,"num_worker":4#nombre de worker, parallélise le chargement de la donnée à l'entrainement
                                   #4 max sur le noeud de dev, pour charger la donnée plus vite, 8 sur les autres noeuds. Mais dépend de la complexité des preprocessing
                    ,"DDP":False#pour la parallélisation de l'entrainement, pas très utile et ne fonctionne pas actuellement (normalement les fonctions sont clean pour les boucles d'entrainements, pas forcément pour les optimiser, loss ....)
                    
                    #param entrainement
                    "lr":base_lr#0.0000001
                    ,"batch_size":8#taille de batch <16 pour un petit GPU c'est pas catastrophique pour notre jeu de donnée, au contraire
                    ,"num_epochs":30
                    ,"optimizer":"ADAM"# "SGD", SGD plus dur a tune
                    ,"momentum":0.9#utile pour SGD, sinon inutilisé

                    #param scheduler
                    ,"scheduler":"onecylce"/"step"/"plateau"
                        #onecycle, learning rate incrémental jusqu'à un max lr, puis décroissant
                            ,'max_lr':base_lr*10#[base_lr/10,base_lr*1,base_lr*10]#une liste avec les learning rate différentiel, un entier sinon
                        #"step"
                            ,"step_size":10 #nombre d'epoch avant decay du learning rate
                        #plateau, le learning rate est diminué tout les "step_size" epoch où la loss n'a pas diminuée
                            ,"gamma":0 #de 0 à 1, valeur de diminution du lr avec lr=lr*gamma
                            ,"step_size":10#nombre d'epoch de patience
                        #

                    #regularisation
                    ,"loss_weights":True/None#pour la donnée imbalanced, calculé par la classe dataset#/!\False peut renvoyer une erreur, None a préférer
                    ,"label_smoothing":0 # entre 0 et 1, ça a été inventé pour géré les mauvaise labellisation, probablement inutile ici
                    ,"L2_reg":0.001#L2 reg par rapports aux poids d'origine, pour le transfert learning
                    ,"weights_decay":True/False#L2 reg pour l'entrainement initial d'un modèle
                    }
"""
def get_training_tools(model,parame):
    """
    récupère les paramètres d'entrainements:
    scheduler,optimizer,loss
    

    """
    #define loss
    criterion = nn.CrossEntropyLoss(weight=parame.get("loss_weights",None),label_smoothing=parame.get("label_smoothing",0))

    #define optimizer
    optimizer=parame.get("optimizer","SGD")
    """
    #this scheme allow for differential learning rate, depend of the architecture
    lr_params=[{'params': model.Feature_extractor[0].parameters(), 'lr': parame.get("lr", 0.001)/100},#pre-conv
                {'params': model.Feature_extractor[1].parameters(), 'lr': parame.get("lr", 0.001)/10},#conv layers
               {'params': model.classifier.parameters(), 'lr': parame.get("lr", 0.001)}]#classifier
    """
    lr_params=model.parameters()
    if optimizer.upper()=="SGD":
        optimizer = optim.SGD(lr_params,
                                lr=parame.get("lr", 0.001), # 0.001 is used if no lr is specified
                                momentum=parame.get("momentum", 0.9),
                                weight_decay=   parame.get("weight_decay", 0),
        )
    elif optimizer.upper()=="ADAM":
        optimizer = optim.Adam(lr_params,
                                lr=parame.get("lr", 0.001), # 0.001 is used if no lr is specified
                             weight_decay=   parame.get("weight_decay", 0),
        )
        
    #define scheduler
    if parame.get("scheduler","Plateau").lower()=="plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=parame.get("gamma", 1.0)
        ,patience=parame.get("step_size", 15))
    elif parame.get("scheduler","Plateau").lower()=="onecycle":
        step_per_epoch=parame["n_sample"]/parame["batch_size"]
        if int(step_per_epoch)!=step_per_epoch:
            step_per_epoch=int(step_per_epoch)+1
        else:
            step_per_epoch=int(step_per_epoch)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,epochs=parame["num_epochs"],steps_per_epoch=step_per_epoch,max_lr=parame["max_lr"])
    elif parame.get("scheduler","Plateau").lower()=="step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(parame.get("step_size", 30)),
            gamma=parame.get("gamma", 1.0),  # default is no learning rate decay
        )
    
    return criterion, optimizer, scheduler

def get_training_tools_LOAD(model,parame):
    """
    récupère les paramètres d'entrainements et charge les poids du modèle:
    scheduler,optimizer,loss
    lorsqu'il y a chargement d'un modèle pour la reprise de l'entrainement, les étapes doivent être bien respecté pour prévenir les bug:
    
    charger le modèle ->GPU
    créer la loss
    créer l'optimizer->charger les poids
    créer le scheduler->charger l'état

    """
    #return the needed object for training
    model_dic=torch.load(parame["model_name_load"])
    model.cuda()#start by loading model on GPU, otherwise mismatch with optimizer parameters
    model.load_state_dict(model_dic["state_dict"])#load model


    criterion = nn.CrossEntropyLoss(weight=parame.get("loss_weights",None),label_smoothing=parame.get("label_smoothing",0))

    #define optimizer
    optimizer=parame.get("optimizer","SGD")
    """
    #this scheme allow for differential learning rate, depend of the architecture
    lr_params=[{'params': model.Feature_extractor[0].parameters(), 'lr': parame.get("lr", 0.001)/100},#pre-conv
                {'params': model.Feature_extractor[1].parameters(), 'lr': parame.get("lr", 0.001)/10},#conv layers
               {'params': model.classifier.parameters(), 'lr': parame.get("lr", 0.001)}]#classifier
    """
    lr_params=model.parameters()
    if optimizer.upper()=="SGD":
        optimizer = optim.SGD(lr_params,
                                lr=parame.get("lr", 0.001), # 0.001 is used if no lr is specified
                                momentum=parame.get("momentum", 0.9),
                                weight_decay=   parame.get("weight_decay", 0),
        )
    elif optimizer.upper()=="ADAM":
        optimizer = optim.Adam(lr_params,
                                lr=parame.get("lr", 0.001), # 0.001 is used if no lr is specified
                             weight_decay=   parame.get("weight_decay", 0),
        )
        
    
    optimizer.load_state_dict(model_dic["optimizer_state_dict"])#load optimizer

    if parame.get("scheduler","Plateau").lower()=="plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=parame.get("gamma", 1.0)
        ,patience=parame.get("step_size", 15))
    elif parame.get("scheduler","Plateau").lower()=="onecycle":
        step_per_epoch=parame["n_sample"]/parame["batch_size"]
        if int(step_per_epoch)!=step_per_epoch:
            step_per_epoch=int(step_per_epoch)+1
        else:
            step_per_epoch=int(step_per_epoch)
        model_dic["scheduler_state_dict"]["total_steps"]=step_per_epoch*parame["num_epochs"]+model_dic["scheduler_state_dict"]["_step_count"]
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,epochs=parame["num_epochs"],steps_per_epoch=step_per_epoch,max_lr=parame["max_lr"])
    elif parame.get("scheduler","Plateau").lower()=="step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(parame.get("step_size", 30)),
            gamma=parame.get("gamma", 1.0),  # default is no learning rate decay
        )
    #update number of total step, isnot properly taken into account otherwise
    scheduler.load_state_dict(model_dic["scheduler_state_dict"])
    
    return criterion, optimizer, scheduler

##train function
class Training:
    
    """
    class for trainning
    
    Define training and evaluation steps,
    
    distributed is implemented but not working on Gricad
    """
    
    def __init__(self,model,training_parameters, dtype):
        """
        

        Parameters
        ----------
        model : torch model
        hyperparameters : dict of hyperparameters
            DESCRIPTION.
        training_parameters : dict
            contains default parameters for training:
                "DDP":distributed
                "save_dic":save_dic
            and eventual hyperparameters if needed to be static for bayesearch
        dtype : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        #param calculs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype=dtype 
        self.num_epochs = training_parameters.get("num_epochs",10)
        self.num_workers = training_parameters.get("num_workers",0)
        self.mixed_precision = training_parameters.get("mixed_precision",False)
        print("Mixed Precision : ",self.mixed_precision)
        self.batch_size = training_parameters.get("batch_size",3)
        self.distributed = training_parameters.get("DDP",False)
        self.save=training_parameters["param_save"].get("save",True)
        self.dic_save=dict(training_parameters["param_save"])
        self.save_best_model=SaveBestModel(training_parameters["param_save"])
        self.weights=[]

        #régularisation
        self.L2_reg=training_parameters.get("L2_reg",0)#regularisation pour le transfert learning
        for p in model.Feature_extractor.parameters():#boucle sur les poids du modèle pour enregistrer l'état initial
            p=p.detach().clone().cuda()
            self.weights+=[p]
        self.total_weights=0
        for i,p in enumerate(model.Feature_extractor.parameters()):#pour obtenir une information sur la valeur par défaut des poids, ajouter un print
            self.total_weights+=((self.weights[i])**2).sum()


        #paramètres d'entrainement
        if training_parameters.get("LOAD",False)==True:
            criterion, optimizer, scheduler = get_training_tools_LOAD(model, training_parameters)
        else:
            criterion, optimizer, scheduler = get_training_tools(model, training_parameters)
        self.criterion=criterion
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.scheduler_name=training_parameters["scheduler"].lower()#l'emplacement de l'accumulateur d'étape dépend du scheduler utilisé
        
    
        self.init_history()#création du dictionnaire pour les courbes d'apprentissage
    
    def init_history(self):
        history = {} # Collects per-epoch loss and acc like Keras' fit().
        history['loss'] = []
        history['val_loss'] = []
        history['acc'] = []
        history['val_acc'] = []
        self.history=history
    
        
    
    @torch.enable_grad()#active le calcul du gradient dans cette fonction
    def train_step(self, model, train_loader, epoch, rank=None):
        """
        entrainement d'une époque
        """
        model.train()
        model.cuda()
        if self.mixed_precision:
            scaler=torch.cuda.amp.GradScaler(enabled=True)#si mixed precision, le scaler compense les défaut de quantization
        train_loss       = 0.0
        num_train_correct  = 0
        num_train_examples = 0
        
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            if self.distributed:
                inputs.cuda(non_blocking=True)
                inputs = inputs.to(dtype=self.dtype)
                inputs = inputs.to(rank)
                labels = labels.cuda(non_blocking=True)
                labels = labels.to(dtype=self.dtype)
                labels = labels.to(rank)
            else:
                inputs = inputs.to(dtype=torch.float, device=self.device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device=self.device)
            if self.mixed_precision==True:#has to be set up that way for some reason #removing == True does not work

                with torch.autocast("cuda",dtype=torch.float16):
                    #outputs = model.Feature_extractor(inputs)
                    #outputs=torch.cat([outputs,labels[:,None]//2],axis=-1)
                    #outputs=model.classifier(outputs)
                    outputs=model(inputs)
                    #labels=labels%2
                    loss = self.criterion(outputs, labels)
                    #print(loss.device)
                    if self.L2_reg>0:
                        sum_=0
                        for i,p in enumerate(model.Feature_extractor.parameters()):
                            sum_+=((p-self.weights[i])**2).sum().cuda()
                        #print("weight decay")
                        #print(sum_)
                        loss+=sum_.cuda()*self.L2_reg
                    #print(loss.device)
                if self.scheduler_name=="onecycle":
                    #étape par batch
                    
                    self.scheduler.step()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()  
            else:
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                if self.L2_reg>0:
                        sum_=0
                        for i,p in enumerate(model.Feature_extractor.parameters()):
                            sum_+=((p-self.weights[i])**2).sum().cuda()
                        #print("weight decay")
                        #print(sum_)
                        loss+=sum_.cuda()*self.L2_reg
                loss.backward()
                self.optimizer.step()
            
            if rank == 0:
                save_model(epoch, model,self.optimizer, self.criterion, self.scheduler)
            #metrics
            train_loss         += loss.data.item() * inputs.size(0)
            num_train_correct  += (torch.max(outputs, 1)[1] == labels).sum().item()
            num_train_examples += inputs.shape[0]
        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_loader.dataset)
        
        self.history['loss'].append(train_loss)
        self.history['acc'].append(train_acc)
        if self.scheduler_name=="step":
            #étape par epoch
            self.scheduler.step()
    
    @torch.no_grad()#désactive le gradient dans cette fonction
    def validation_step(self,model,val_loader,rank,other_scores=False):
        """
        étape d'évaluation, toute la donnée de validation y est vue.
        par défaut l'accuracy est celle calculé,

        l'option other_score est utile seulement pour obtenir d'autres métriques d'évaluations dans l'appel externe:
            il n'y a pas d'incrémentation du dictionnaire des métriques au cours de l'entrainement
            pas hyper propre mais évite de redéfinir une fonction
        """
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0
        y_pred=[]
        y_true=[]
        for i, (images, labels) in enumerate(val_loader):

            if self.distributed:
                images = images.cuda(non_blocking=True)
                images = images.to(self.dtype)
                images = images.to(rank)
                labels = labels.cuda(non_blocking=True)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(rank)
            else:
                images = images.to(dtype=self.dtype, device=self.device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device=self.device)
            outputs=model(images)
            loss = self.criterion(outputs, labels)

            val_loss         += loss.data.item() * images.size(0)
            if other_scores:
                y_pred+=torch.max(outputs, 1)[1].detach().tolist()
                y_true+=labels.detach().tolist()
            else:
                num_val_correct  += (torch.max(outputs, 1)[1] == labels).sum().item()
                num_val_examples += images.shape[0]
        if other_scores:
            return f1_score_(y_true, y_pred, average="weighted"), balanced_accuracy_score(y_true,y_pred)
        else:
            val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_loader.dataset)
        if self.scheduler_name=="plateau":
            
            #étape par epoch,dépend de la loss de validation
            self.scheduler.step(val_loss)
        
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        
    
    def train_loop(self,model,train_loader,val_loader=None,
             rank=None,verbose=True,save_best_model=False):
        """
        

        Parameters
        ----------
        model : cannot be set as an attribute for DDP
            trainning data
        train_loader : torch loader
            trainning data
        val_loader : torch loader, optional
            validation data or test data regarding the needs
        rank : int, optional
            arguments for distributed computing on GPU
        verbose : Bool, optional
            same as keras
        save_best_model : Bool, optional
            save the model at each epoch if validation result is better than the previous one
                "model_name_best_model.pth"
            also enable default save, the model is saved at each epoch
                "model_name_final_model.pth"

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
              (type(model).__name__, type(self.optimizer).__name__,
               self.optimizer.param_groups[0]['lr'], self.num_epochs, self.device))
            
        model.to(dtype=self.dtype, device=self.device)
          
            
        start_time_sec = time.time()    
        for epoch in range(self.num_epochs):
            self.train_step(model, train_loader, epoch, rank)
            if val_loader is not None :
                self.validation_step(model, val_loader,rank)
                if save_best_model:
                    self.save_best_model(self.history["val_loss"][-1], epoch
                                     , model, self.optimizer, self.criterion, self.scheduler)
                else:
                    assert not save_best_model, "saving while training need a validation set to compare models"
            if save_best_model:
                #print("default save, overwrite")
                save_model(range(1,epoch), model, self.optimizer, self.criterion, self.scheduler,self.dic_save)
            if verbose or self.history["val_loss"][-1]==np.min(self.history["val_loss"]):#si verbose ou meilleure loss, afficher logs
                if val_loader is not None:
                    print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                        (epoch, self.num_epochs, self.history["loss"][-1], self.history["acc"][-1]
                         , self.history["val_loss"][-1], self.history["val_acc"][-1]))
                else:
                    print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f' % \
                        (epoch, self.num_epochs, self.history["loss"][-1], self.history["acc"][-1]))
        if verbose==False:
            if val_loader is not None:
                print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                    (epoch, self.num_epochs, self.history["loss"][-1], self.history["acc"][-1]
                        , self.history["val_loss"][-1], self.history["val_acc"][-1]))
            else:
                print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f' % \
                    (epoch, self.num_epochs, self.history["loss"][-1], self.history["acc"][-1]))
        
        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.num_epochs
        print()
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
        return model
    
    def evaluate(self,model,val_set,rank=None,other_scores=False):
        """
        fonction d'évaluation, jamais utilisé en dehors du gridsearch

        model:modèle a évaluer
        val_set:dataset pour le calcul des métrique
        rank:pour l'entrainement distribué (paramètre automatique)
        other_score:pour utiliser d'autre score que l'accuracy, utile pour le gridsearch, cf validation_step pour ces métriques
        """
        n_gpus = torch.cuda.device_count()
        if n_gpus >=2 and self.distributed:
            self.evaluate_DDP(model,val_set)
        else:
            test_loader = torch.utils.data.DataLoader(val_set,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     pin_memory=False)

            
        if other_scores:
            self.history["val_loss"].pop(-1), self.history["val_acc"].pop(-1)
            return self.validation_step(model,test_loader,rank,other_scores=other_scores)
        #empty last frame from dict
        self.validation_step(model,test_loader,rank,other_scores=other_scores)
        return self.history["val_loss"].pop(-1), self.history["val_acc"].pop(-1)
    

    def fit(self,model,train_set,val_set=None,verbose=True,save_best_model=False):
        """
        fonction de base pour commencer l'entrainement

        Parameters
        ----------
        model : model pytorch à entrainer

        trainning data
        train_set : torch dataset
            trainning data
        val_set : torch dataset or None, optional
            validation data
            dans un gridsearch on préférera l'option None, si possible, pour éviter de révaluer le modèle à chaque époch

        verbose : Bool, optional
            pareil que keras, permet un affichage exhaustif ou non des logs
        save_best_model : Bool, optional
            enregistre le modèle sous la forme "model_name_best_model.pth" si le modèle est meilleur 
            selon les métrique de validation(loss) que l'époch précédente
            active aussi la sauvegarde par défaut: le modèle est enregistré à chaque époque sous la forme "model_name_final_model.pth"
        
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        n_gpus = torch.cuda.device_count()
        if n_gpus >=2 and self.distributed:
            print(f"computing using {n_gpus} GPUs")
            return self.fit_DDP(model,train_set,val_set)
        else:
            train_loader = torch.utils.data.DataLoader(train_set,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         num_workers=self.num_workers,
                                         pin_memory=False)
            if val_set:
                test_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            num_workers=self.num_workers,
                                            pin_memory=False)
            else:
                test_loader=None
            model.to(dtype=self.dtype, device=self.device)
            return self.train_loop(model,train_loader,test_loader
                                   ,verbose=verbose,save_best_model=save_best_model)

    def fit_DDP(self,model,train_set,val_set):#distributed
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_demo(fntls.partial(_fit_ddp_main,self,model,train_set,val_set), world_size)
        pass
    
    def evaluate_DDP(self,model,test_set):#distributed
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_demo(lambda x,y:self.evaluate_ddp_main(x,y,model, test_set), world_size)
        
    

        
    
    def fit_ddp_main(self,model,train_set,val_set, rank, world_size):#distributed
        #rank = dist.get_rank()
        rank = rank % torch.cuda.device_count()
        print("rank : ",rank)
        print("world_size : ",world_size)
        
        #model=self.model
        #train_set=self.train_set
        #val_set=self.test_set
        #verbose=False
        # setup the process groups
        setup(rank, world_size)
        # prepare the dataloader
        train_loader = prepare(rank, world_size, train_set, batch_size=self.batch_size)
        val_loader = prepare(rank, world_size, val_set, batch_size=self.batch_size)
        
        # instantiate the model(it's your own model) and move it to the right device
        print("loader initialisated")
        modelddp = model.to(rank)
        modelddp = modelddp.to(self.dtype)

        print("model initialisated")
        # wrap the model with DDP
        # device_ids tell DDP where is your model
        # output_device tells DDP where to output, in our case, it is rank
        # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
        
        modelddp = DDP(modelddp, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        self.train_loop(modelddp,train_loader,val_loader,rank=rank,verbose=True,
                        save_best_model=False)
        
        cleanup()
    
    def evaluate_ddp_main(self, rank, world_size, model, val_set):#distributed
        
        # setup the process groups
        setup(rank, world_size)
        # prepare the dataloader
        val_loader = prepare(rank, world_size, val_set, batch_size=self.batch_size)
        
        # instantiate the model(it's your own model) and move it to the right device

        modelddp = model.to(rank)
        modelddp = modelddp.to(self.dtype)
        
        # wrap the model with DDP
        # device_ids tell DDP where is your model
        # output_device tells DDP where to output, in our case, it is rank
        # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
        
        modelddp = DDP(modelddp, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        self.eval_step(modelddp,val_loader,rank=rank)
        
        cleanup()
        
    
    

def GS_train_evaluate(parameterization, train_set, val_set, model, parameters_training_static
                   , trial_count, metric="acc", dtype=torch.float):
    """
        
    fonction utilisé par le bayesian search

    Parameters
    ----------
    parameterization: dict,
        paramètres à explorer, obtenue par la selection selon la bayesiansearch
    train_set : torch dataset
        trainning data
    val_loader : torch dataset
        validation data or test data regarding the needs
    model: torch model
    metric: acc, balanced_acc, f1_score_weightened
    parameters_training_static : dict, optional
        paramètres fixés, pas a explorer

    Returns
    score d'évaluation
    -------
    TYPE
        DESCRIPTION.

    """
    num_trial=next(trial_count)
    _model = model()#instantiate model at each trial

    training_parameters =  dict(parameterization)#copy le dictionnaire, pour éviter l'effet de bord

    duplicate_keys=[k for k in training_parameters if k not in parameters_training_static]#vérficiation des clefs en double
    if duplicate_keys:#si il existe des clefs en double, liste non vide<=>True
        print("/!\ paramètres à la fois utilisé pour l'exploration et fixe, le paramètre est fixé par défaut dans ce cas")
    training_parameters.update(parameters_training_static)#fusionne les dictionnaires de paramètres /!\ les paramètres staqtique écrase les paramètre non fixe
    
    if parameters_training_static.get("finetuning",False):
        _model.FT(training_parameters)
    Train=Training(_model, training_parameters, dtype)#TODO dic save
    
    _model=Train.fit(_model, train_set, val_set, save_best_model=False,verbose=False) #fit model
    if metric=="acc":
        loss,acc=Train.evaluate(_model,val_set,other_score=False)#get metrics
        score=acc
    else:
        balanced_acc,f1_score=Train.evaluate(_model,val_set,other_score=True)#get metrics
        if metric=="f1_score":
            score=f1_score
        elif metric=="balanced_acc":
            score=balanced_acc
    
    return score


from sklearn.model_selection import StratifiedKFold



def GS_CV_train_evaluate(parameterization, model
                   , parameters_training_static
                   , trial_count, subject_ID_list, file_list, label_list
                   , FlattenData
                   , metric="acc"
                   , dtype=torch.float):
    """
        
    fonction utilisé par le bayesian search avec cross validation
    pertinant seulement avec la donnée flatten

    Parameters
    ----------
    parameterization: dict,
        paramètres à explorer, obtenue par la selection selon la bayesiansearch
    parameters_training_static : dict
        paramètres fixés, pas a explorer
    model: torch model

    
    metric: acc, balanced_acc,f1_score_weightened
    trial_count: iterable pour avoir accès au numéro de run.
    subject_ID_list : liste des ID de sujets pour le split
    file_list : liste des échantillons (sequence IRMf)
    label_list : list des labels
    Flatten_Data : torch dataset pour générer les échantillons flatten
    Returns
    score d'évaluation
    -------
    TYPE
        DESCRIPTION.

    """
    training_parameters =  dict(parameterization)
    training_parameters.update(parameters_training_static)
    print(training_parameters)

    subject_ID_list = np.array(subject_ID_list)
    CV_gen=CV_generator(file_list,subject_ID_list,n_fold=5)
    score_kfold=[]
    for train_index, test_index in CV_gen:#récupère itérativement les indexes des groupes de plis
        _model = model()#instanciation du modèle à chaque itération
        
        if parameters_training_static.get("finetuning",False):
            _model.FT(training_parameters)
        #récupération des fichiers,labels du split
        X_train=np.array(file_list)[train_index]
        y_train=np.array(label_list)[train_index]
        X_test=np.array(file_list)[test_index]
        y_test=np.array(label_list)[test_index]

        train_set = FlattenData(X_train,y_train,is_train=True)#création du dataset d'entrainement
        val_set = FlattenData(X_test,y_test,is_train=False)#création du dataset d'évaluation

        Train=Training(_model, training_parameters, dtype)
    
        _model=Train.fit(_model, train_set, val_set, save_best_model=False,verbose=False) 
        if metric=="acc":
            loss,acc=Train.evaluate(_model,val_set,other_score=False)#get metrics
            score=acc
        else:
            balanced_acc,f1_score=Train.evaluate(_model,val_set,other_score=True)#get metrics
            if metric=="f1_score":
                score=f1_score
            elif metric=="balanced_acc":
                score=balanced_acc
    
        score_kfold+=[score]
    acc=np.mean(score_kfold)
    print(f"end_loop {metric} :",score)
    return score



#DDP


def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    dataset = dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, sampler=sampler)

    return dataloader


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("NCCL", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    


def _fit_ddp_main(Trainer,model,train_set,val_set, rank, world_size):
    return Trainer.fit_ddp_main(model,train_set,val_set, rank, world_size)