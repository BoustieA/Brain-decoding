# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:38:25 2022

@author: adywi
"""

import os, fnmatch
import numpy as np
import nibabel as nib

#path = os.getcwd()

##Set directory
# Change the current working directory

path="DATA_HCP/test_code/"
read_path = os.path.abspath(path+"DATA_RAW")
save_path = os.path.abspath(path+"DATA_transformed_max")

##Select file for right-to-left scans for emotion

file_list = []

for path, folders, files in os.walk(read_path):
    for file in files:
        if fnmatch.fnmatch(file, '*nii.gz'):
            file_list.append(file)

print(len(file_list))

for file in file_list:
    img = nib.load(os.path.join(read_path,file))
    data = img.get_fdata()
    data = data[8:-8, 8:-8, :-10, :27]
    data = data / data.max(axis=3)[:, :, :, np.newaxis]
    data[~ np.isfinite(data)] = 0
    filename = file[:-6] + ".npy"
    np.save(os.path.join(save_path,file), data)
