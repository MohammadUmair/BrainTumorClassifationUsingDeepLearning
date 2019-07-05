# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:43:06 2019

@author: Umair
"""

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
filename = get_testdata_files("CT_small.dcm")[0]
ds = pydicom.dcmread(filename)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 

location = "E:\\2 Education\\BSSE\\7th Semester\\FYP\\Maaz\\MRI\\Brain-Tumor-Progression\\PGBM-001\\11-19-1991-FH-HEADBrain Protocols-40993\\5364-MaskTumor-94345\\000000.dcm"
location2 = "E:\\2 Education\\BSSE\\7th Semester\\FYP\\Maaz\\MRI\Brain-Tumor-Progression\\PGBM-001\\11-19-1991-FH-HEADBrain Protocols-40993\\11-T1post-03326\\000000.dcm"
d = pydicom.dcmread(location)
plt.imshow(d.pixel_array, cmap=plt.cm.bone) 
d.pixel_array.shape

d1 = pydicom.dcmread(location2)
plt.imshow(d1.pixel_array, cmap=plt.cm.bone) 
d1.pixel_array.shape

location3 = "E:\\2 Education\\BSSE\\7th Semester\\FYP\\Maaz\\MRI\\Brain-Tumor-Progression\\PGBM-001\\11-19-1991-FH-HEADBrain Protocols-40993\\37910-T2reg-84816\\000000.dcm"
d2 = pydicom.dcmread(location3)
plt.imshow(d2.pixel_array, cmap=plt.cm.bone) 
d2.pixel_array.shape


location4 = "E:\\2 Education\\BSSE\\7th Semester\\FYP\\Maaz\\MRI\\Brain-Tumor-Progression\\PGBM-001\\11-19-1991-FH-HEADBrain Protocols-40993\\35004-dT1-72693\\000018.dcm"
d3 = pydicom.dcmread(location4)
plt.imshow(d3.pixel_array, cmap=plt.cm.bone) 
d3.pixel_array.shape


location5 = "E:\\2 Education\\BSSE\\7th Semester\\FYP\\Maaz\\MRI\\Brain-Tumor-Progression\\PGBM-002\\01-17-1997-MR RCBV SEQUENCE-36058\\34909-T1prereg-84610\\000001.dcm"
d4 = pydicom.dcmread(location5)
plt.imshow(d4.pixel_array, cmap=plt.cm.bone) 
d4.pixel_array.shape


data = d
fig, ax = plt.subplots(1, 1, figsize=[4, 5])
n = 0
ax[n].imshow(data[:, :],'gray')
ax[n].set_xticks([])
ax[n].set_yticks([])
ax[n].set_title('Slice number: {}'.format(slice), color='r')
n += 1
fig.subplots_adjust(wspace=0, hspace=0)
plt.show()

