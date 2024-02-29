
"""
Created on Thu Oct 26 11:23:47 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""

import nibabel as nib
import torch
from torch.nn import ReLU
import os
import sys
import glob
import numpy as np
import pandas as pd
import gc

from Models.resnet18 import S3ConvXFCResnet
from Models.DeepBrain_wang import DeepBrain
from utilities.Dataset import get_torch_datasets
from nilearn.datasets import fetch_surf_fsaverage

import matplotlib.pyplot as plt
from Models.guided_backprop import GuidedBackprop
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from nilearn.image import mean_img
from nilearn.image import concat_imgs
from nilearn import plotting
import torch
from torch import nn
from torch.functional import F
import nibabel as nib
from nilearn import plotting
from nilearn.image import mean_img,index_img,resample_img
from nilearn.datasets import load_mni152_template, fetch_neurovault_motor_task
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = False
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
torch.cuda.empty_cache()
from scipy.ndimage import gaussian_filter, median_filter


def get_border_volume(img_,template=False):
    
    """
    calcul le plus petit rectangle contenant les voxels non nuls
    """
    if template:
        img=img_>np.mean(img_)
    else:
        img=img_>np.mean(img_)
    for i in range(img.shape[0]):
        if img[i,:,:].max()>0:
            min_left=i
            break
    for i in range(img.shape[0]-1,-1,-1):
        if img[i,:,:].max()>0:
            min_right=i
            break
        
    for i in range(img.shape[1]):
        if img[:,i,:].max()>0:
            min_top=i
            break
    for i in range(img.shape[1]-1,-1,-1):
        if img[:,i,:].max()>0:
            min_bottom=i
            break
        
    for i in range(img.shape[2]):
        if img[:,:,i].max()>0:
            min_back=i
            break
    for i in range(img.shape[2]-1,-1,-1):
        if img[:,:,i].max()>0:
            min_front=i+1
            break
    return min_left, min_right,    min_top,    min_bottom,min_back,min_front

def resize_frame(brain_file,GBP_file,template,frame):
    """
    Met à l'echelle les images pour correspondre aux dimensions du template
    Calcul les paramètres de mise à l'echelle en fonction de l'image IRMf de base
    et les utilise dans la mise à l'echelle de l'image obtenue par guided backpropagation

    brain_file:IRMf
    GBP_file:résultat guided backpropagation
    template:MNII template

    """
    slice_=3
    frame_t_template=template.get_fdata()#récupère le template au format npy
    a,b,c,d,e,f=get_border_volume(frame_t_template,True)#calcul le plus petit rectangle contenant le cerveau du template
    frame_t_template=frame_t_template[a:b,c:d,e:f]#extrait ce rectangle

    
    brain_file=brain_file-brain_file.min()
    brain_file=brain_file/brain_file.max()#normalise l'IRMF
    frame_t=brain_file[frame]#récupère la frame correspondante
    mask=frame_t>0#mask pour supprimer les artefacts de la GBP
    print("frame to resize")
    print(frame_t.max())
    print(frame_t.min())
    a_,b_,c_,d_,e_,f_=get_border_volume(frame_t,False)#calcul le plus petit rectangle de l'IRMF
 
    frame_GBP=(GBP_file*mask)[a_:b_,c_:d_,e_:f_]#coupe l'image GBP selon le rectangle prédéfini
    
    frame_t_GBP=resize(frame_GBP,frame_t_template.shape)#redimensionne par interpolation selon les dimensions du template
    
    resized=np.zeros(template.shape)
    resized[a:b,c:d,e:f]=frame_t_GBP#padding pour obtenir la taille du 
    """
    frame_t=resize(frame_t[a_:b_,c_:d_,e_:f_],frame_t_template.shape)#resize brain file as well
    resized_brain=np.zeros(template.shape)
    resized_brain[a:b,c:d,e:f]=frame_t
    """
    m=np.min(resized)
    M=np.max(resized)
    M=np.max([np.abs(m),np.abs(M)])
    resized=resized/M
    
    resized_file=nib.Nifti1Image(resized,template.affine)#convertit au format nii
    return resized_file

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        #first_layer = list(self.model.preBlock._modules.items())[0][1]#wang model
        first_layer = list(self.model.Feature_extractor[0]._modules.items())[0][1]#resnet
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
#         for pos, module in self.model.features._modules.items():
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(input_image.device)
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.cpu().numpy()[0]
        return gradients_as_arr


group="phono"
label=1
print("#"*10)
print(device)
print("plot GBP features")
path="./Records/trained_models/"
model_name=f"Finetuning_Resnet_INLANG_Task_vs_Control_{group}_8_best_model.pth"#'checkpoint_24.pth.tar'#
#model_name="DeepBrain_Global_norm_2_suite_suite_final_model.pth"
#model_name='checkpoint_24.pth.tar'
param_data={"DATA":"INLANG_CONTROL",
            "processing":"oversample",
            "env":"silenus",
            "split":"final_test",
            "len_seq":14,
            "normalization":"global",
            "crop":False}

num_class=7


def visualise(param_data, model, device):
    #template = load_mni152_template()
    num_class=2
    feature_list = []
    device = device
    dataset, _ = get_torch_datasets(**param_data)
    dataset.file_list=dataset.file_list#[:256]
    dataset.label_list=dataset.label_list#[:256]
    """
    for i,l in enumerate(dataset.label_list):
        if l==1:
            index=i
            break
    
    dataset.file_list=[dataset.file_list[index]][:1]#
    dataset.label_list=[dataset.label_list[index]][:1]
    """
    for sample in iter(dataset):
        shape=sample[0].shape
        break
    #prepare empty array to sum up results (average)
    dic_result_by_label={i :np.zeros(shape) for i in range(num_class)}#{i :torch.zeros(shape).to(device=device) for i in range(num_class)}
    dic_brain_by_label={i :np.zeros(shape) for i in range(num_class)}#{i :torch.zeros(shape).to(device=device) for i in range(num_class)}#to resize
    dic_count_by_label={i:0 for i in range(num_class)}
    dic_prob_by_label={i:0 for i in range(num_class)}
    dataset_loader = torch.utils.data.DataLoader(dataset,
                            batch_size=1,#batch_size=parameterization.get("batchsize", 3),
                            shuffle=False,
                            num_workers=0)
    path="./Records/trained_models/"
    #W=weights of each model
    W=[0.65, 0.875, 0.95, 0.575, 0.95, 0.85, 0.65, 0.6, 0.6, 1.0]#
    W=[0.825,0.9,0.95,0.625,1,0.825,0.725,0.875,0.7,1]
    for num_model in range(10):
        model_name=f"Finetuning_Resnet_INLANG_Task_vs_Control_{group}_{num_model}_best_model.pth"#'checkpoint_24.pth.tar'#

        #init architecture at each step, is cleared at each run to empty space
        model=S3ConvXFCResnet(27,num_class)
        #model=DeepBrain()
        #adapt classifier to the one saved
        if group=="sem":
            model.classifier=nn.Sequential(nn.Dropout(0.25),
            nn.Linear(512,2))
        elif group=="phono":
            model.classifier=nn.Sequential(nn.Linear(512,2))

        
        model.load_state_dict(torch.load(path+model_name)['state_dict'])#load model
        model.to(device)
        model.eval()#set up model in eval mode, disable gradient update
        GBP = GuidedBackprop(model)
        for i,sample in enumerate(dataset_loader):
            img, label = sample
            img = img.to(dtype=torch.float32, device=device)
            label = label.item()
            
            with torch.no_grad():#calculate the accuracy for this sample
                pred_prob = F.softmax(model(img))[0]
            
            print("true label :",label)
            print("predict_proba :", pred_prob)
            
            # Remove LogSoftmax# check GBP publication, the information is specific to one class only

            prob=pred_prob[label].item()
            dic_prob_by_label[label]+=(prob)*W[num_model]
            input_img = img.to(device)
            input_img = nn.Parameter(input_img, requires_grad=True).to(device)
            
            #calculate GBP image
            guided_grads = GBP.generate_gradients(input_img, label)
            
            guided_grads=guided_grads*(guided_grads>0)#remove negative values of first layer
            guided_grads=guided_grads*W[num_model]*prob#weightened the result

            #add the results to compute the mean
            dic_brain_by_label[label]+=img[0].detach().cpu().numpy()   
            dic_result_by_label[label]+=guided_grads#.detach().cpu().numpy()
            dic_count_by_label[label]+=1
        
            #prevent bug
            if i%8==0:#this aim to reset cache of intermediate output of previous image likely the cause of memory error
                del model
                del img
                del GBP
                del input_img
                del sample
                del label
                del guided_grads
                del pred_prob
                gc.collect()
                torch.cuda.empty_cache()
                #redefine model architecture
                model=S3ConvXFCResnet(27,num_class)
                #model=DeepBrain()
                if group=="sem":
                    model.classifier=nn.Sequential(nn.Dropout(0.25),
                nn.Linear(512,2))
                elif group=="phono":
                    model.classifier=nn.Sequential(nn.Linear(512,2))
                model.zero_grad()
                model.load_state_dict(torch.load(path+model_name)['state_dict'])
                model.to(device)
                model.eval()
                GBP = GuidedBackprop(model)
                print(i," clean")
        #compute the mean
        for i in dic_result_by_label:
            if dic_count_by_label[i]!=0:#prevent bug
                dic_result_by_label[i]=dic_result_by_label[i]/dic_count_by_label[i]/dic_prob_by_label[i]
        for i in dic_brain_by_label:
            if dic_count_by_label[i]!=0:#prevent bug
                dic_brain_by_label[i]=dic_brain_by_label[i]/dic_count_by_label[i]#/dic_prob_by_label[i]
    
    return dic_result_by_label, dic_brain_by_label

param_data={"DATA":"INLANG_CONTROL",
            "processing":"oversample",
            "env":"silenus",
            "split":"final_test",
            "len_seq":14,
            "normalization":"global",
            "crop":False}

template=load_mni152_template(resolution=2.2)

dic_grad, dic_brain = visualise(param_data, model, device)
path="./Records/GBP_plot/"+group+"/"
dic_grad={k:np.mean(dic_grad[k],axis=0) for k in dic_grad}#moyenne les frames
for j in range(2):
    if j==0:
        var="task"
    else:
        var="control"
    
    #make the plot 
    brain_file=dic_brain[j] 
    GBP_file =dic_grad[j]
    resized_file=resize_frame(brain_file,GBP_file,template,0)
    
    #path="./Records/GBP_plot"
    print(resized_file.get_fdata().max())
    print(resized_file.get_fdata().min())
    print(resized_file.get_fdata().shape)

    plotting.plot_stat_map(resized_file,template,threshold=0.35
    )#,cut_coords=(14,6, 0))
    path_image=os.path.join(path,f'{group}_{i}_{var}_stat_map_average_w.png')
    plt.savefig(path_image)
    plt.close()


    #fsaverage = fetch_surf_fsaverage()
    plotting.plot_glass_brain(resized_file,threshold=0.35)
    path_image=os.path.join(path,f'{group}_{i}_{var}_glass_brain_average_w.png')
    plt.savefig(path_image)
    plt.close()



    plotting.plot_img_on_surf(resized_file ,threshold= 0.35
        , inflate='True'
        , views=['lateral', 'medial'],   hemispheres=['left', 'right']
        , colorbar=True, title='Cartographie de surface\n des activations', surf_mesh='fsaverage')
    path_image=os.path.join(path,f'{group}_{i}_{var}_surface_average_w.png')
    plt.savefig(path_image)
    plt.close()
        