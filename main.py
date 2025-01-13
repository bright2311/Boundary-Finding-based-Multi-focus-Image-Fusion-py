# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 23:16:52 2025

@author: Bright
"""
import os
import cv2
import numpy as np
from boundary_region_finding import bondary_extract
from boundary_reconstruct import boundary_line_extraction,boundary_line_adjustment
from boundary_fusion import boundary_fusion
from utils import showImg

if __name__=="__main__":
    
    p1="img/clock1.bmp"
    p2="img/clock2.bmp"

    img1=cv2.imread(p1,-1) 
    img2=cv2.imread(p2,-1)
    rgb=len(img1.shape)==3
    if rgb:
        img1=cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
        img2=cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)
        h,w=img1.shape[:2]
        img1=cv2.resize(img1,(w//4,h//4))
        img2=cv2.resize(img2,(w//4,h//4))

    sw_sz = 7 # search window size
    scales = 5 # weighted focus-measure scales
    b_sz = 20  # boundary refinement in a local boundary region
        
    decision_map,FM1,FM2=bondary_extract(img1, img2, sw_sz, scales, b_sz)
    
    decision_map=boundary_line_extraction(decision_map)
    # showImg(decision_map,"showImg3")
    decision_map= boundary_line_adjustment(img1, img2, decision_map,FM1,FM2,scales, b_sz)
    # showImg(decision_map,"decision_map")
    # print(np.unique(decision_map))
    # boundary fusion, with size sz
    if rgb:
        img1=cv2.imread(p1,-1) 
        img2=cv2.imread(p2,-1)
        img1=cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
        img2=cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
        h,w=img1.shape[:2]
        img1=cv2.resize(img1,(w//4,h//4))
        img2=cv2.resize(img2,(w//4,h//4))
    
    bf_sz = 3    
    # showImg(img1,"img1")
    # showImg(img2,"img2")
    fusion=boundary_fusion(img1, img2, decision_map, bf_sz)
    cv2.imwrite("fusion.jpg",fusion)
    showImg(fusion,"showImg4")
