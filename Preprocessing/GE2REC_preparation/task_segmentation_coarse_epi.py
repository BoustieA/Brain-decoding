#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:34:40 2022

@author: neurodeep
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:12:12 2022

@author: neurodeep
"""

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


path = r"/media/neurodeep/My Book/LTLE"

masked_list = []
gene_list = []
rap_list = []

for path, folders, files in os.walk(path):
    for file in files:
        if fnmatch.fnmatch(file, '*masked.nii'):
            masked_list.append(os.path.join(path, file))
            
            
for file in masked_list:
    if fnmatch.fnmatch(file, '*180dyn*'):
        gene_list.append(os.path.join(path, file))


for file in masked_list:
    if fnmatch.fnmatch(file, '*100dyn*'):
        rap_list.append(os.path.join(path, file))



for file in gene_list:
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    filename1 = file[:-10] + "1_gene_train_epi.nii"
    filename2 = file[:-10] + "2_gene_train_epi.nii"
    filename3 = file[:-10] + "3_gene_train_epi.nii"
    filename4 = file[:-10] + "4_gene_train_epi.nii"
    filename5 = file[:-10] + "5_gene_train_epi.nii"
    img = nib.load(file)
    data1 = img.slicer[:,:,:,0:16]
    data2 = img.slicer[:,:,:,5:16]
    data3 = img.slicer[:,:,:,36:52]
    data4 = img.slicer[:,:,:,41:52]
    data5 = img.slicer[:,:,:,72:88]
    data6 = img.slicer[:,:,:,77:88]
    data7 = img.slicer[:,:,:,108:124]
    data8 = img.slicer[:,:,:,113:124]
    data9 = img.slicer[:,:,:,144:160]
    data10 = img.slicer[:,:,:,149:160]
    l1.append(data1)
    l1.append(data2)
    l2.append(data3)
    l2.append(data4)
    l3.append(data5)
    l3.append(data6)
    l4.append(data7)
    l4.append(data8)
    l5.append(data9)
    l5.append(data10)
    fmri1 = concat_imgs(l1)
    fmri2 = concat_imgs(l2)
    fmri3 = concat_imgs(l3)
    fmri4 = concat_imgs(l4)
    fmri5 = concat_imgs(l5)
    fmri1.to_filename(filename1)
    fmri2.to_filename(filename2)
    fmri3.to_filename(filename3)
    fmri4.to_filename(filename4)
    fmri5.to_filename(filename5)
    
    
for file in rap_list:
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    filename1 = file[:-10] + "1_rap_train_epi.nii"
    filename2 = file[:-10] + "2_rap_train_epi.nii"
    filename3 = file[:-10] + "3_rap_train_epi.nii"
    filename4 = file[:-10] + "4_rap_train_epi.nii"
    filename5 = file[:-10] + "5_rap_train_epi.nii"
    img = nib.load(file)    
    data1 = img.slicer[:,:,:,0:16]
    data2 = img.slicer[:,:,:,5:16]
    data3 = img.slicer[:,:,:,20:36]
    data4 = img.slicer[:,:,:,25:36]
    data5 = img.slicer[:,:,:,40:56]
    data6 = img.slicer[:,:,:,45:56]
    data7 = img.slicer[:,:,:,60:76]
    data8 = img.slicer[:,:,:,65:76]
    data9 = img.slicer[:,:,:,80:96]
    data10 = img.slicer[:,:,:,85:96]
    l1.append(data1)
    l1.append(data2)
    l2.append(data3)
    l2.append(data4)
    l3.append(data5)
    l3.append(data6)
    l4.append(data7)
    l4.append(data8)
    l5.append(data9)
    l5.append(data10)
    fmri1 = concat_imgs(l1)
    fmri2 = concat_imgs(l2)
    fmri3 = concat_imgs(l3)
    fmri4 = concat_imgs(l4)
    fmri5 = concat_imgs(l5)
    fmri1.to_filename(filename1)
    fmri2.to_filename(filename2)
    fmri3.to_filename(filename3)
    fmri4.to_filename(filename4)
    fmri5.to_filename(filename5)