#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:05:39 2022

@author: neurodeep
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:53:16 2022

@author: neurodeep
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:20:50 2022

@author: neurodeep
"""

import fnmatch
import os
import numpy as np
from nilearn.image import resample_to_img as res
import nibabel as nib
import nilearn
from  nilearn.maskers import NiftiLabelsMasker
from  nilearn.maskers import NiftiMasker
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
from nilearn.masking import unmask
import cv2
from nilearn.image import iter_img
from nilearn.image import math_img
from nilearn.image import concat_imgs
from nilearn.image import crop_img


example = nib.load('/media/neurodeep/My Book/HCP/AllRepeatsCutAndLabel/100206_EMOTION_LR-1_fear.nii.gz')
example =  example.slicer[8:-8, 8:-8, :-10, :]

path = r"/media/neurodeep/My Book/new_pretreatment_task_rs"

nii_list = []

for path, folders, files in os.walk(path):
    for file in files:
        if fnmatch.fnmatch(file, '*.nii'):
            nii_list.append(os.path.join(path, file))
            
gene_list = []
rap_list = []

for file in nii_list:
    if fnmatch.fnmatch(file, '*GENE/swar*'):
        gene_list.append(os.path.join(path, file))

for file in nii_list:
    if fnmatch.fnmatch(file, '*RAPPEL/swar*'):
        rap_list.append(os.path.join(path, file))
        
len('/media/neurodeep/My Book/new_pretreatment_task_rs/SEMVIE-01/GENE/warSEMVIE-01_Plast-FAIS_Gene_180dyn_LL_3_1.nii')
        
gene_list.sort()
rap_list.sort()

for (f) in gene_list:
    fmri = nib.load(f)
    mask = nilearn.masking.compute_epi_mask(fmri)
    filename = f[:-17] + "masked.nii"
    masked_list = []
    
    for img in iter_img(fmri):
        masked = math_img('img*mask', img = img, mask=mask)
        masked_list.append(masked)
        
    final = concat_imgs(masked_list)
    crop = crop_img(final)
    print(crop.shape)
    crop_resampled = res(crop, example, interpolation = 'nearest')
    print(crop_resampled.shape)  
    crop_resampled.to_filename(filename)    


for (f) in rap_list:
    fmri = nib.load(f)
    mask = nilearn.masking.compute_epi_mask(fmri)
    filename = f[:-17] + "masked.nii"
    masked_list = []
    
    for img in iter_img(fmri):
        masked = math_img('img*mask', img = img, mask=mask)
        masked_list.append(masked)
        
    final = concat_imgs(masked_list)
    crop = crop_img(final)
    print(crop.shape)
    crop_resampled = res(crop, example, interpolation = 'nearest')
    print(crop_resampled.shape)  
    crop_resampled.to_filename(filename)  


