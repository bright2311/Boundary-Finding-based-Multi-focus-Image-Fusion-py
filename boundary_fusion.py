# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:25:21 2025

@author: Bright
"""
import numpy as np
import cv2
from utils import kernels,showImg


def boundary_fusion(img1, img2, dfmap, sz):
    # boundary_fusion fuses two source images accoring to decision map

    # define the size of structuring element
    SE = np.array(kernels[2*sz-1], dtype=np.uint8).reshape(2*sz-1, 2*sz-1)
    map_erode = cv2.morphologyEx(dfmap.astype(np.uint8), cv2.MORPH_ERODE, SE)
    # for item in np.unique(map_erode):
    #     showImg(map_erode==item,f"{item}")
    # showImg(map_erode,"map_erode")

    # focused regions
    map1 = (dfmap == 1).astype(np.uint8)
    map2 = (dfmap == 2).astype(np.uint8)
    breg = (dfmap == 0).astype(np.uint8)
    # showImg(breg,"breg")

    # distance transform
    eps= np.finfo(np.float32).eps#1e-2#

    map1_n = cv2.bitwise_not(map1.astype(np.uint8)*255)
    map2_n = cv2.bitwise_not(map2.astype(np.uint8)*255)
    dt1 = cv2.distanceTransform(map1_n, cv2.DIST_L2, 3)+ eps
    dt2 = cv2.distanceTransform(map2_n, cv2.DIST_L2, 3)+ eps
    # showImg(dt1==dt2,"dt1==dt2")

   # distance sum
    dt = dt1 + dt2

    # distance weight
    coeff1 = dt2/ dt#(dt + eps)
    coeff2 = dt1 / dt#(dt + eps)


    # Boundary region
    if len(img1.shape)==3:
        map12=np.zeros_like(img1)
        map22=np.zeros_like(img2)
        coeff12=np.zeros_like(img1)
        coeff22=np.zeros_like(img2)
        breg2=np.zeros_like(img2)
        for i in range(map12.shape[-1]):
            map12[...,i]=map1
            map22[...,i]=map2
            coeff12[...,i]=coeff1
            coeff22[...,i]=coeff2
            breg2[...,i]=breg

        # Initial Image
        fimg = (map12 * img1 + map22 * img2).astype(np.uint8)
        # boundary region fusion image
        bimg = ((img1 * coeff12 + img2 * coeff22) * breg2).astype(np.uint8)
        # bimg = ((img1 * 0.5 + img2 * 0.5) * breg2).astype(np.uint8)

    else:
        # Initial Image
        fimg = (map1 * img1 + map2 * img2).astype(np.uint8)
        # showImg(fimg,"fimg")
        # boundary region fusion image
        bimg = ((img1 * coeff1 + img2 * coeff2) * breg).astype(np.uint8)
        showImg(bimg,"bimg")
    # final fusion image
    fimg += bimg
    
    return fimg
