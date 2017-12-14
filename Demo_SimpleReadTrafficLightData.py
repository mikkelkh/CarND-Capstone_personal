#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:33:45 2017

@author: pistol
"""
import re
import os
import glob
import numpy as np
from shutil import copyfile

dirIn = "TrafficLight_Session0"

# Renaming 
#files = glob.glob(os.path.join(dirIn,"*.jpg"))
#for sFile in files:
#    strParts = re.split('[/ .]',sFile)
#    
#    sFileOut = os.path.join(dirOut,strParts[1]+'.'+strParts[3])
#    copyfile(sFile,sFileOut)

label_dist_list = []
image_list = []
files = glob.glob(os.path.join(dirIn,"*.jpg"))
sFile = files[0]
for sFile in files:
    str_parts = re.split('[/ .]',sFile)[1]
    label_dist_list.append(map(int,str_parts.split('_')[1:]))
    image_list.append(sFile)
    
    
label_dist = np.asarray(label_dist_list)



images = image_list
labels = label_dist[:,0]
dist = label_dist[:,1]