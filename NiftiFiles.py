# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:32:16 2019

@author: Umair
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:23:46 2019

@author: Umair
"""

import nibabel as nib
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
#import cv2

a=nib.load('E:\\Umair12\\Desktop\\testing\\random.nii')
a=a.get_data()
a.shape


a=nib.load("E:\\2 Education\\BSSE\\7th Semester\\FYP\\Current Working\\abc\\IXI079-HH-1388-IXIMADisoTFE12_-s3T129_-0401-00004-000001-01.nii")
a=a.get_data()
a.shape

a=nib.load("E:\\2 Education\\BSSE\\7th Semester\\FYP\\Brain\\Neurohacking_data-0.0\\BRAINIX\\NIfTI\\BRAINIX_NIFTI_FLAIR.nii")
a=a.get_data()
a.shape


data = a

#fmri=nib.load('E:\\Umair12\\Downloads\\Compressed\\MoAEpilot\\sub-01\\func\\sub-01_task-auditory_bold.nii')
#fmri = fmri.get_data()
#fmri.shape

#smri = nib.load('E:\\Umair12\\Downloads\\Compressed\\MoAEpilot\\sub-01\\anat\\sub-01_T1w.nii')
#smri = smri.get_data()
#smri.shape

a.shape
#v = a
#x = 192 * 256 * 192 * 1 
#print(x)
#a.reshape(np.prod(a[:-1],a.shape[-1]))



#data = np.rot90(a.squeeze(), 1) #k=1 Number of times the array is rotated by 90 degrees.
                                   #npArray.squeeze() The input array, but with all or a subset of the dimensions of length 1 removed. This is always a itself or a view into arr. 
#print(data.shape)                  

     
fig, ax = plt.subplots(3, 1, figsize=[18, 9])
n = 0
slice = 10
for _ in range(3):
    ax[n].imshow(data[:, :, slice],'gray')
    ax[n].set_xticks([])
    ax[n].set_yticks([])
    ax[n].set_title('Slice number: {}'.format(slice), color='r')
    n += 1
    slice += 1
    fig.subplots_adjust(wspace=0, hspace=0)
plt.show()



fig, ax = plt.subplots(30, 1, figsize=[180, 90])
n = 0
slice = 30
for _ in range(30):
    ax[n].imshow(data[slice, :, :], 'gray')
    ax[n].set_xticks([])
    ax[n].set_yticks([])
    ax[n].set_title('Slice number: {}'.format(slice), color='r')
    n += 1
    slice += 1
    
fig.subplots_adjust(wspace=0, hspace=0)
plt.show()