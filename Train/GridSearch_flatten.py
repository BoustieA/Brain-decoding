# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 20:57:55 2023

@author: 33695
"""
print("#"*10)
print("start Grid search")
import fnmatch
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from torch.utils.data import Dataset

#from utilities.PATH import get_torch_datasets
from utilities.Dataset import get_dic_path, get_torch_datasets
from sklearn.utils import shuffle

#personnal script

from utilities.model_description import get_n_parameters


from Models.DeepBrain import DeepBrain
from Models.resnet18 import S3ConvXFCResnet
from Models.Attention_net_wang import Attention_network
from utilities.train_func_NN import GS_CV_train_evaluate

from utilities.Dataset import FlattenData


torch.backends.cudnn.benchmark=True
print(torch.cuda.is_available())
t0 = time.time()
torch.cuda.empty_cache()
dtype=torch.float


#PARAMETERS for single run

#model name for saving
model_name_save="test_GS"
SAVING=False
#model_type
model_architecture="classifier_flatten_inlang"#"DeepBrain"

#data choice

#FT param
finetuning=False


#parameters training
#ce sont les paramètres exploré par le gridsearch,
parameters_training_gs=[
    {"name": "lr", "type": "range", "bounds": [1e-6, 0.05], "log_scale": True},
    {"name": "weight_decay", "type": "range", "bounds": [1e-6, 0.01], "log_scale": True},
    {"name": "batchsize", "type": "range", "bounds": [2, 128]},
    {"name": "step_size", "type": "range", "bounds": [2, 40]},
    #{"name": "layer_size","type": "choice", "values": [64, 128, 256],"value_type":"int"},
    #{"name": "optimizer","type":"choice", "values": ["ADAM","SGD"],"value_type":"str"},
    #{"name": "activation","type":"choice", "values": ["relu","sigmoid"],"value_type":"str"},
    {"name": "drop_out","type": "range", "bounds": [0.01,0.5]},
    {"name": "gamma", "type": "range", "bounds": [0.1, 0.99]}]


#parameters training static are the parameters needed for training with no search space
#The GS_loop extract a dic from the searched one and is then fused with statics one
parameters_training_static={"DDP":False,
                            "n_output":3,#Fine tuning output# unused if finetuning to False
                            "finetuning":True,
                            "num_epochs":50,
                            "activation":"relu",
                            "layer_size":64,
                            "optimizer":"ADAM",
                            "momentum":0.5}


#GS trials#number of test
total_trial=20




#Automatic script



#load path and update dic with previous manual parameters
dic_path=get_dic_path()
dic_path.update({"model_architecture":model_architecture,
                "model_name":model_name_save,})
parameters_training_static["param_save"] = dic_path
parameters_training_static["param_save"]["save"]=SAVING





#dataset #get the files at hand
print("cur dir ",os.path.abspath(os.curdir))
path_data="../Data/InLang/Innerspeech_flatten_pytorch_label2/"
path_data='../../../../../silenus/PROJECTS/pr-deepneuro/COMMON/DATA/INLANG/INNERSPEECH/FLATTEN_CONCATENATED/'
files=[]
labels=[]
subject_ID_list=[]
files_name=os.listdir(path_data)
files_name.sort()
for file in files_name:
    if fnmatch.fnmatch(file, '*.npy'):
        files+=[np.load(path_data+file)]
        labels+=[int(file[-5])]#/!\ depend of the extension
        subject_ID_list+=[file[:3]]#/!\depend of the string, might start with warsub
files_name=[path_data+f for f in files_name]
subject_ID_list=np.unique(subject_ID_list)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



print("subject ID for train",subject_ID_list)
#define architecture, do not instantiate ! it is done inside the CV loop to initialise the weights
model=CF
def trial_count(n=0):
    while n>=0:
        n=n+1
        yield n







trial_counter=trial_count()

train_evaluate=lambda x: GS_CV_train_evaluate(x, model, parameters_training_static
                   , trial_count, subject_ID_list, files_name, labels
                   , FlattenData
                   , metric="balanced_acc"
                   , dtype=torch.float)

#optimization loop over search parameters
best_parameters, values, experiment, model = optimize(
    parameters=parameters_training_gs,

    total_trials=total_trial,
    evaluation_function = train_evaluate,
    objective_name='accuracy',
)
 


print(best_parameters)
means, covariances = values
print(means)
print(covariances)





#Plot accuracy

best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])

best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Classification Accuracy, %",
)



render(best_objective_plot)

#find best hyper parameter

data = experiment.fetch_data()
df = data.df
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
best_arm = experiment.arms_by_name[best_arm_name]
print(best_arm)

best_param=best_arm._parameters
print(best_param)
#record results
with open(model_name_save+"_best_param.txt","w") as f:
    f.write(json.dumps(best_param))
#save
