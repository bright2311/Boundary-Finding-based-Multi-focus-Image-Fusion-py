# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 19:23:08 2025

@author: Bright
"""
import cv2
import numpy as np
from utils import show_subPlots, showImg, kernels
from boundary_region_finding import multiscale_morph, decision_map_detection

from skimage.segmentation import watershed

def bwareaopen(BW, P, conn):
    # delete small connectivity (smaller than P) from bitwise map
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        BW, connectivity=conn)

    mask = np.zeros_like(labels, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < P:
            mask[labels == i] = 255
    mask = cv2.bitwise_not(mask)
    BW = cv2.bitwise_and(BW, BW, mask=mask)

    return BW


def Small_Block_Filter(decision_map, N, small_size):
    #  small_size is the defined number for small regions
    #  mp is the initial fusion decision map;
    #  N is the number of the input images

    # define the size of the small region

    P = small_size
    conn = 4
    if N == 2:
        map1 = (decision_map == 1)
        map2 = (decision_map == 2)

        # Process the Positive image, delete the small patches
        tmap1 = bwareaopen(map1.astype(np.uint8), P, conn)
        tmap2 = bwareaopen(map2.astype(np.uint8), P, conn)

        titles = ["map1", "tmap1", "map2", "tmap2"]
        mats = [map1, tmap1, map2, tmap2]
        show_subPlots(mats, titles, 2, 2, False)

        # sum map
        sum_map = (tmap1 & tmap2)  # useless?
        # showImg(sum_map,"sum_map")

        # the final decision map
        newmap = (tmap1 + tmap2 * 2) * (1 - sum_map)
        # showImg(newmap,"newmap")
    else:
        h, w = decision_map.shape[:2]
        result = np.zeros((h, w, N), dtype=np.uint8)
        for ii in range(N):
            ptmap = (decision_map == ii)
            result[..., ii] = bwareaopen(ptmap, P, conn)

        # Find if there are confused pixels
        sCount = np.sum(result, axis=-1)
        Tag = (sCount == 1)

        newmap = np.zeros_like(decision_map)
        for ii in range(1, N+1):
            newmap += ii*result[..., ii]*Tag
    return newmap


def Nearest_Filter(decision_map, N):
    yy, xx = decision_map.shape[:2]
    tmap = (decision_map == 0).astype(np.int8)
    # showImg(tmap, "tmap")
    # Process the Positive image, delete the small patches
    num_labels, L, stats, _ = cv2.connectedComponentsWithStats(
        tmap, connectivity=8)
    # showImg(L, "L")
    # for each non regions, find the its bounding box
    tL = np.zeros((yy, xx), dtype=np.int8)

    for ii in range(1, num_labels):
        x, y, w, h = stats[ii, :4]
        # print(ii,x,y,w,h)

        # Large scale boundingbox
        left = max(0, x - 2)
        right = min(x + w+1, xx)

        top = max(0, y - 2)
        bottom = min(y + h+1, yy)

        # This method is only true for two input images
       # Extract the block bounding from the map
        region = decision_map[top: bottom, left: right]

        # Count the number of the regions in the bounding box
        numCount1 = np.sum(region == 1)
        numCount2 = np.sum(region == 2)

        if numCount1 > numCount2:
            tL[L == ii] = 1
        elif numCount1 <= numCount2:
            tL[L == ii] = 2

    return decision_map.astype(np.int8) + tL * tmap


def refine_boundary(decision_map, Boundary):
    # -return the refined decision map without isolated boundary lines.
    # Initialize the refined decision map
    refined_decision_map = decision_map.copy()
    # Boundary=cv2.bitwise_not(Boundary)
    # find the boundary positions
    row, col = np.where(Boundary > 0)
    row_num = len(row)
    box_sz = 3

    pad_dfmap = cv2.copyMakeBorder(decision_map,
                                   box_sz//2, box_sz//2, box_sz//2, box_sz//2, cv2.BORDER_REFLECT)
    # print("pad_dfmap:",np.unique(pad_dfmap))
    h, w = decision_map.shape[:2]



    # count and filter the isolated boundaries
    for ii in range(row_num):
        x, y = row[ii], col[ii]
        box_data = pad_dfmap[x: x + box_sz, y: y + box_sz]
        tag = 0
        # # maximum number filter
        isempty1 = ((box_data == 1).sum() == 0)
        isempty2 = ((box_data == 2).sum() == 0)

        if(not isempty1) & isempty2:
            tag = 1
        elif isempty1 & (not isempty2):
            tag = 2

        refined_decision_map[x, y] = tag
    # decision_map_color = np.zeros((h, w, 3), dtype=np.uint8)
    # decision_map_color[decision_map == 0] = (255, 0, 0)
    # decision_map_color[decision_map == 1] = (0, 255, 0)
    # decision_map_color[decision_map == 2] = (255, 255, 255)
    # showImg(decision_map_color, "decision_map_color")
    # cv2.imwrite("decision_map_color.jpg",decision_map_color)

    return refined_decision_map


import skimage.morphology as morph

def imposemin(img, minima):
    '''
    https://stackoverflow.com/questions/70251361/imextendedmin-and-imimposemin-functions-in-python-for-watershed-seeds-from-dista
    '''
    marker = np.full(img.shape, np.inf)
    marker[minima == 1] = 0
    mask = np.minimum((img + 1), marker)
    return morph.reconstruction(marker, mask, method='erosion')


def maker_watershed(image, markers, opt):
    # maker_watershed utilizes marker based watershed algorithm to adjust boundaries

    if opt == "Prewitt":
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

        dx = cv2.filter2D(image, cv2.CV_16S, kernelx)
        dy = cv2.filter2D(image, cv2.CV_16S, kernely)
        # print("dx:",dx.shape,dx)
        # print("dy:",dy.shape,dy)
        absX=cv2.convertScaleAbs(dx)
        absY=cv2.convertScaleAbs(dy)
        prewitt_map = cv2.addWeighted(absX,0.5,absY,0.5,0)
        # prewitt_map = np.sqrt(dx**2 + dy**2).astype(np.uint8)
    # showImg(prewitt_map,"prewitt_map0")

    prewitt_map=imposemin(prewitt_map, markers)
    # showImg(prewitt_map,"prewitt_map")
    # mask=prewitt_map>0

    labels = watershed(prewitt_map, connectivity=2,watershed_line=True)
    # showImg(labels,"labels")
    # print(np.unique(labels))
    result= (labels==0).astype(np.uint8)*255
    # showImg(result,"watershed")
    return result

    # return res

def fusion_image(img1, img2,decision_map):
    img2 = img2.astype(float)

    map1 = (decision_map == 1).astype(float)
    map2 = (decision_map == 2).astype(float)
    map3 = (decision_map < 1).astype(float)

    fimg = map1 * img1 + map2 * img2 + map3 * (img1 + img2) / 2
    return fimg.astype(np.uint8)


def boundary_line_extraction(decision_map):
    Boundary = (decision_map == 0)
    # showImg(decision_map, "decision_map0")
    # showImg(Boundary, "Boundary")

    p1, p2 = Boundary.shape[:2]
    block_sz = round(p1 * p2 / 40)#  100#
    # print("decision_map0: ",np.unique(decision_map),np.unique(Boundary))

    decision_map = Small_Block_Filter(decision_map, 2, block_sz)
    Boundary_numeric = np.zeros_like(Boundary, dtype=int)
    Boundary_numeric[Boundary > 0] = 1
    # showImg(decision_map, "decision_map1")
    # showImg(Boundary_numeric, "Boundary")

    decision_map = decision_map.astype(int) - Boundary_numeric
    # showImg(decision_map, "decision_map2")

    decision_map = Nearest_Filter(decision_map, 2)
    # showImg(decision_map, "decision_map3")
    # print("decision_map3: ",np.unique(decision_map))

    decision_map0 = refine_boundary(decision_map, Boundary_numeric)
    # showImg(decision_map0, "decision_map4")
    return decision_map0


def boundary_line_adjustment(img1, img2, decision_map, FM1, FM2, scale_num, b_sz, opt='Prewitt'):
    
    SE = np.array(kernels[2*b_sz-1],dtype=np.uint8).reshape(2*b_sz-1, 2*b_sz-1)
    # SE = np.ones((39,39), np.uint8)

    adjustment_region = cv2.morphologyEx(decision_map.astype(np.uint8), cv2.MORPH_ERODE, SE)
    # showImg(adjustment_region,"adjustment_region")


    adjustment_region = (adjustment_region > 0)
    # showImg(adjustment_region,"adjustment_region")
    FM1=FM1.astype(np.uint8)
    FM2=FM2.astype(np.uint8)
    # showImg(FM1,"FM1")
    # showImg(FM2,"FM2")

    ws1 = maker_watershed(FM1, adjustment_region, opt)
    ws2 = maker_watershed(FM2, adjustment_region, opt)

    ws1_n = cv2.bitwise_not(ws1)
    ws2_n = cv2.bitwise_not(ws2)

    # Regenerate the decision maps for other boundaries
    decision_map1 = decision_map_detection(FM1, FM2, ws1_n,  4,False)
    # showImg(decision_map1,"showImg31")
    decision_map2 = decision_map_detection(FM1, FM2, ws2_n, 4)
    
    # showImg(decision_map2,"showImg32")

    # To select a best boundary by comparing of the fused images
    fimg1 = fusion_image(img1, img2, decision_map1)
    fimg2 = fusion_image(img1, img2, decision_map2)
    # showImg(fimg1,"fimg1")
    # showImg(fimg2,"fimg2")

    # Compare the MS-FM
    F_FM1 = np.mean(multiscale_morph(fimg1, scale_num))
    F_FM2 = np.mean(multiscale_morph(fimg2, scale_num))
    # print(F_FM1,F_FM2)

    # image with high focus-measure is selected as the final fused image
    if F_FM1 > F_FM2:
        # ws = ws1
        decision_map = decision_map1
    else:
        # ws = ws2
        decision_map = decision_map2
    # showImg(decision_map,"showImg33")

    #Refine the boundaries in the decision map
    decision_map=boundary_line_extraction(decision_map)
  
    return decision_map
