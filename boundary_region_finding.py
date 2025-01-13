# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 23:22:41 2025

@author: Bright
"""

import cv2
import numpy as np
from utils import multiscale_morph,show_subPlots,showImg

def remove_short_line(thin):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    # showImg(labels>0,"labels")
    # print(np.unique(labels))
    area_num = []
    for i in range(1, num_labels): 
        area = stats[i, cv2.CC_STAT_AREA]
        area_num.append(area)
        
    area_sort=sorted(area_num,reverse=True)
    # print(area_sort)
    line_num=int(np.ceil(num_labels* 0.2))
    large_area = area_sort[:line_num]
    th = np.mean(large_area)
    ppL = np.zeros_like(labels,dtype=np.uint8)
    for i in range(1,num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= th:
            ppL[labels == i] = 255
    labels=cv2.bitwise_not(ppL).astype(np.uint8)
    return labels


def decision_map_detection(FM1,FM2,L,Conn,show=False):

    num_labels, Conn_Reg, stats, _ = cv2.connectedComponentsWithStats(L, connectivity=Conn)
    # print(np.unique(Conn_Reg),num_labels)
    # decision_map = Conn_Reg#.copy()
    decision_map=np.zeros_like(Conn_Reg,dtype=np.uint8)
    

    for ii in range(1, num_labels): 
        tag = (Conn_Reg == ii)

        label = 0
        #Compute the focus measure of this region
        sumFM1 = np.sum(FM1 * tag)
        sumFM2 = np.sum(FM2 * tag)
        if show:
            print(ii,sumFM1,sumFM2)
            showImg(tag,"tag")        
            
        if sumFM2 > sumFM1:
            label = 2
        elif sumFM2 < sumFM1:
            label = 1
        
        # Define this region
        decision_map[tag] = label
    return decision_map

def bondary_extract(img1, img2, sw_sz, scales, b_sz):
    # Compute saliency for each image
    scale_num = scales
    FM1 = multiscale_morph(img1, scale_num)
    FM2 = multiscale_morph(img2, scale_num)

    # Sum of the focus-measure
    H = np.ones((sw_sz, sw_sz))
    sumFM1 = cv2.filter2D(FM1, -1, H, borderType=cv2.BORDER_CONSTANT)
    sumFM2 = cv2.filter2D(FM2, -1, H, borderType=cv2.BORDER_CONSTANT)
    maxFM = cv2.max(FM1, FM2)
    #minFM = cv2.min(FM1, FM2)

    # Max and min of the sum of the focus-measure
    max_sumFM = cv2.max(sumFM1, sumFM2)
    min_sumFM = cv2.min(sumFM1, sumFM2)
    # detect the boundary regions
    sum_maxFM = cv2.filter2D(maxFM, -1, H, borderType=cv2.BORDER_CONSTANT)
    #sum_minFM = cv2.filter2D(minFM, -1, H, borderType=cv2.BORDER_CONSTANT)

    sum_minFM = sumFM1 + sumFM2 - sum_maxFM
    # print(sum_minFM)
    # maximum difference in focus-measures (MDFM)
    dif_max_min_sum = max_sumFM - min_sumFM

    # the sum of the maximum difference in gradients (SMDG)
    dif_sum_max_min = sum_maxFM - sum_minFM

    dfmap = np.array((dif_max_min_sum >= 0.8 *
                      dif_sum_max_min) & (dif_max_min_sum >= sw_sz**2), 
                     dtype=np.uint8)*255
    p1,p2=img1.shape[:2]
    # large_dfmap = np.zeros((p1 + 4, p2 + 4),dtype=np.uint8)
    # large_dfmap[2 :  - 2, 2 :  - 2]= dfmap
    cv2.bitwise_not(dfmap, dfmap)
    thin = cv2.ximgproc.thinning(dfmap)
    # thin = thin[2 :  - 2, 2 :  - 2]
    L=remove_short_line(thin)
        
    decision_map=decision_map_detection(FM1,FM2,L,4)

    mats=[FM1,FM2,dif_max_min_sum,dif_sum_max_min,dfmap,thin,L,decision_map]
    titles=["FM1","FM2","MDFM","SMDG","dfmap","thin","L","decision_map"]

    # (decision_map==1).sum(),decision_map==1).sum()

    show_subPlots(mats,titles,4,2,False)
    # showImg(decision_map, "decision_map")
    
    # h,w=decision_map.shape[:2]
    # decision_map_color=np.zeros((h,w,3),dtype=np.uint8)
    # decision_map_color[decision_map==0]=(0,0,0)
    # decision_map_color[decision_map==1]=(0,255,0)
    # decision_map_color[decision_map==2]=(255,0,0)
    # showImg(decision_map_color, "decision_map_color")

    return decision_map,FM1,FM2



    
