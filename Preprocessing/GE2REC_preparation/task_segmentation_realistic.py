#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:44:53 2022

@author: neurodeep
"""

import fnmatch
import os
import numpy as np
import nibabel as nib
from nilearn.image import concat_imgs


path = r"/media/neurodeep/My Book/new_pretreatment_task_rs"

masked_list = []
gene_list = []
rap_list = []

for path, folders, files in os.walk(path):
    for file in files:
        if fnmatch.fnmatch(file, '*masked.nii'):
            masked_list.append(os.path.join(path, file))
            
            
for file in masked_list:
    if fnmatch.fnmatch(file, '*GENE/war*'):
        gene_list.append(os.path.join(path, file))

for file in masked_list:
    if fnmatch.fnmatch(file, '*RAPPEL/war*'):
        rap_list.append(os.path.join(path, file))


for file in gene_list:
    l1 = []
    l2 = []
    filename1 = file[:-9] + "1_gene_segmented.nii"
    filename2 = file[:-9] + "2_gene_segmented.nii"
    img = nib.load(file)
    data1 = img.slicer[:,:,:,36:52]
    data2 = img.slicer[:,:,:,77:88]
    data3 = img.slicer[:,:,:,108:124]
    data4 = img.slicer[:,:,:,149:160]
    l1.append(data1)
    l1.append(data2)
    l2.append(data3)
    l2.append(data4)
    fmri1 = concat_imgs(l1)
    fmri2 = concat_imgs(l2)
    fmri1.to_filename(filename1)
    fmri2.to_filename(filename2)
    
for file in rap_list:
    l1 = []
    l2 = []
    filename1 = file[:-9] + "1_rap_segmented.nii"
    filename2 = file[:-9] + "2_rap_segmented.nii"
    img = nib.load(file)
    data1 = img.slicer[:,:,:,20:36]
    data2 = img.slicer[:,:,:,45:56]
    data3 = img.slicer[:,:,:,60:76]
    data4 = img.slicer[:,:,:,85:96]
    l1.append(data1)
    l1.append(data2)
    l2.append(data3)
    l2.append(data4)
    fmri1 = concat_imgs(l1)
    fmri2 = concat_imgs(l2)
    fmri1.to_filename(filename1)
    fmri2.to_filename(filename2)

