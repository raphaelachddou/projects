#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:27:30 2017

@author: raphael
"""
import numpy as np
from PIL import Image
import cv2
import math
from cv2.ximgproc import guidedFilter
import os 


def guidedfilter(im1,im2,r,epsilon): #implementation originale (n'est plus utilisée)
   
    
    mat1 = np.array(im1, np.float32)
    mat2 = np.array(im2, np.float32)

    n,p = mat1.shape
    A = np.zeros((n,p), np.float32)
    B = np.zeros((n,p), np.float32)

    AB = mat1*mat2 # produit terme à terme de mat1 et mat2
    
    
    for i in range(n):

        i0 = max(0, i-r)
        i1 = min(i+r, n)

        for j in range(p):

            j0 = max(0, j-r)
            j1 = min(j+r, p)

            mean1 = np.mean(mat1[i0:i1, j0:j1])
            mean2 = np.mean(mat2[i0:i1, j0:j1])
            var1 =  np.var(mat1[i0:i1, j0:j1])

            w=(i1-i0+1)*(j1-j0+1)
            
            A[i,j] = (np.sum(AB[i0:i1, j0:j1])/w - mean1 * mean2)/(var1+epsilon)
            
            B[i,j] = mean2 - A[i,j]*mean1

    O = np.zeros((n,p), np.float32)
    print("deuxième étape")
    for i in range(n):

        i0 = max(0, i-r)
        i1 = min(i+r, n)
        for j in range(p):
            j0 = max(0, j-r)
            j1 = min(j+r, p)
            meanA = np.mean(A[i0:i1, j0:j1])
            meanB = np.mean(B[i0:i1, j0:j1])
            O[i,j] = meanA*mat1[i,j] +meanB

    return(O)



def guidedfilterbis(im1,im2,r,epsilon): #pour les images en couleurs

    n,p  = im2.shape

    im1 = im1/255 #Les pixels peuvent prendre des valeurs entre 0 et 1
    im2 = im2/255

    ksize = (2*r+1, 2*r+1)
    kernel = np.ones(ksize) # Noyau de convolution, représente le patch de taille 2*r+1
    
    W = cv2.filter2D(np.ones((n,p)), -1, kernel, borderType = cv2.BORDER_ISOLATED) # Pour normaliser les moyennes tout en prenant en compte les bords

    mean_1r = cv2.filter2D(im1[:,:,0], -1,kernel, borderType = cv2.BORDER_ISOLATED) / W #on calcule les moyennes de chaque patch pour chaque couleur
    mean_1g = cv2.filter2D(im1[:,:,1], -1,kernel, borderType = cv2.BORDER_ISOLATED) / W
    mean_1b = cv2.filter2D(im1[:,:,2], -1,kernel, borderType = cv2.BORDER_ISOLATED) / W


    mean_2 = cv2.filter2D(im2, -1,kernel, borderType = cv2.BORDER_ISOLATED) / W #moyennes par patch pour l'image 2(en niveau de gris)

    mean_12r = cv2.filter2D(im1[:,:,0]*im2, -1,kernel, borderType = cv2.BORDER_ISOLATED) / W #on calcule les moyennes par patch du produit de chaque canal couleur de l'image 1 avec la deuxième 
    mean_12g = cv2.filter2D(im1[:,:,1]*im2, -1,kernel, borderType = cv2.BORDER_ISOLATED) / W
    mean_12b = cv2.filter2D(im1[:,:,2]*im2, -1,kernel, borderType = cv2.BORDER_ISOLATED) / W


    cov_12r = mean_12r - mean_1r * mean_2 #la covariance d'un canal couleur de l'image 1 et l'image2
    cov_12b = mean_12b - mean_1b * mean_2 
    cov_12g = mean_12g - mean_1g * mean_2
    cov_12 = np.array([cov_12r, cov_12b, cov_12g])


    var_1rr = cv2.filter2D(im1[:,:,0]*im1[:,:,0], -1,kernel, borderType = cv2.BORDER_ISOLATED) / W - mean_1r * mean_1r # Pour le calcul de la matrice de covariance de l'image 1
    var_1rg = cv2.filter2D(im1[:,:,0]*im1[:,:,1], -1,kernel, borderType = cv2.BORDER_ISOLATED) / W - mean_1r * mean_1g
    var_1rb = cv2.filter2D(im1[:,:,0]*im1[:,:,2], -1,kernel, borderType = cv2.BORDER_ISOLATED) / W - mean_1r * mean_1b
    var_1gg = cv2.filter2D(im1[:,:,1]*im1[:,:,1], -1,kernel, borderType = cv2.BORDER_ISOLATED) / W - mean_1g * mean_1g
    var_1gb = cv2.filter2D(im1[:,:,1]*im1[:,:,2], -1,kernel, borderType = cv2.BORDER_ISOLATED) / W - mean_1g * mean_1b
    var_1bb = cv2.filter2D(im1[:,:,2]*im1[:,:,2], -1,kernel, borderType = cv2.BORDER_ISOLATED) / W - mean_1b * mean_1b


    a = np.zeros((n,p,3))
    for i in range(n):
        for j in range(p):

            sigma = np.array([[var_1rr[i,j],var_1rg[i,j],var_1rb[i,j] ],[var_1rg[i,j],var_1gg[i,j],var_1gb[i,j] ],[var_1rb[i,j],var_1gb[i,j],var_1bb[i,j] ]]) # matrice de covariance

            eps = epsilon*np.identity(3)

            a[i,j,:] = np.dot((sigma+eps), cov_12[:,i,j]) # Equation (7) de l'article

    
    b = mean_2 - a[:, :, 0] * mean_1r - a[:, :, 1] * mean_1g - a[:, :, 2] * mean_1b # Equation (8) de l'article
    

   

    o = (cv2.filter2D(a[:, :, 0], -1, kernel, borderType = cv2.BORDER_ISOLATED)* im1[:, :, 0]
        + cv2.filter2D(a[:, :, 1], -1, kernel, borderType = cv2.BORDER_ISOLATED)* im1[:, :, 1]
        + cv2.filter2D(a[:, :, 2], -1, kernel, borderType = cv2.BORDER_ISOLATED)* im1[:, :, 2]
        + cv2.filter2D(b, -1, kernel, borderType = cv2.BORDER_ISOLATED))/W  # Equation (9) de l'article
    return(o)

def guidedfilter0(im1,im2,r,epsilon): # pour les images en niveau de gris

    n,p  = im2.shape

    im1 = im1/255 #Les pixels peuvent prendre des valeurs entre 0 et 1
    im2 = im2/255

    ksize = (2*r+1, 2*r+1)
    kernel = np.ones(ksize) # Noyau de convolution, représente le patch de taille 2*r+1
    
    W = cv2.filter2D(np.ones((n,p)), -1, kernel, borderType = cv2.BORDER_ISOLATED) # Pour normaliser les moyennes tout en prenant en compte les bords

    mean_1 = cv2.filter2D(im1, -1,kernel, borderType = cv2.BORDER_ISOLATED) / W #on calcule les moyennes de chaque patch
    
    mean_2 = cv2.filter2D(im2, -1,kernel, borderType = cv2.BORDER_ISOLATED) / W #moyennes par patch pour l'image 2(en niveau de gris)

    mean_12 = cv2.filter2D(im1*im2, -1,kernel, borderType = cv2.BORDER_ISOLATED) / W #on calcule les moyennes par patch du produit de l'image 1 avec la deuxième 


    cov_12 = mean_12 - mean_1 * mean_2 #la covariance de l'image 1 et l'image2


    var_1 = cv2.filter2D(im1*im1, -1,kernel, borderType = cv2.BORDER_ISOLATED) / W - mean_1 * mean_1 # variance de chaque patch de l'image 1

    a = cov_12/(np.ones((n,p))+var_1) #equation (3) de l'article
    
    b = mean_2 - a*mean_1 # Equation (4) de l'article

    o = (cv2.filter2D(a, -1, kernel, borderType = cv2.BORDER_ISOLATED)* im1
        + cv2.filter2D(b, -1, kernel, borderType = cv2.BORDER_ISOLATED))/W  # Equation (9) de l'article
    return(o)

def baseLayer(mat1):
    

    kernel = 1/(32**2)*np.ones((32,32))
    B = cv2.filter2D(mat1, -1, kernel) # filtre moyen de taille (32, 32)

    return(B)

def weightMap(imaList):
    
    
    if len(imaList[0].shape) !=2 :
        grayList=[]
        for im in imaList:
            grayList.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) #transformation de l'image en niveau de gris
    
    else :
        grayList = imaList[:]

    kernel = np.array([[0, -1, 0], [-1, 4, -1] ,[0, -1, 0]]) # noyau de convloution, filtre laplacien

    sal =[]
    print(len(grayList))
    i=1
    for im in grayList:
        m = cv2.filter2D(im, -1, kernel)
        m = cv2.GaussianBlur(np.abs(m), (11, 11), 5,5) #filtre gaussion de taille 11 X 11 et de sigma=5
        
        sal.append(m)
        i+=1


    sal = np.array(sal)
    n,p = sal[0,:,:].shape
    ### Construction des matrices de poids ###
    for i in range(n):
        for j in range(p):
            L=[im[i,j] for im in sal] #liste des pixels i,j de chaque image de la liste
            index = L.index(max(L))  # recherche de l'indice du max de cette liste
            sal[index, i, j] = 255  # celui-ci devient un point blanc
            L[index]=255
            for im in sal:
                if L.index(im[i,j]) != index:
                    im[i,j]=0. # on donne la valeur 0 à tous les autres pixels

    return(sal)


def fusion(imageList1, imageList2):

    l = len(imageList1)
    n,p = imageList2[0].shape
    fusion = imageList1[0][:,:]
    for i in range(n):
        for j in range(p):
            sum=0
            for k in range(l):
                sum+=imageList1[k][i,j]*imageList2[k][i,j]

            fusion[i,j]=sum
    

    return(fusion)



def execute(string, fusionname):

    color = 1
    
    st =os.listdir(string)# lecture des images à partir d'un dossier

    print(st[0]) #st[0] n'est pas un nom d'image donc on l'enlève de la liste à l'étape suivante

    if (st[0]=='.DS_Store'): #dans certains dossiers, st contient une chaine '.DS_Store', nous la supprimons
        imList = st[1:]
    else : 
        imList = st
    ### on prend souvent la précaution de copier la liste originale en utilisant [:] pour éviter de la modifier à chaque modification de la liste copiée

    l = len(imList)
    for i in range(l) :
        imList[i] = cv2.imread(string+'/'+imList[i]) #lecture des images du dossier

    if len(imList[0].shape)==2:
        color = 0 # séparation du cas où l'image est en couleur ou en niveau de gris
    

    imList1 = [imList[i]/255 for i in range(len(imList))]
    baseLayerList = imList[:]
    detailLayerList = []
    WB = imList[:]
    WD = imList[:] #création de listes de même taille et même type que la liste originale pour les modifier facilement

    print('baseLayer')
    for i in range(l): 
        baseLayerList[i] = baseLayer(imList1[i])
        detailLayerList.append(imList1[i]-baseLayerList[i])
        
    
    print('weightMap')
    weightMapList = weightMap(imList[:])


    imList2 = imList[:]
    weightMapList1 = weightMapList[:]

    if color ==1 :
        for i in range(len(imList)):
            print('guidedfilter 1')
            pic = imList2[i][:,:]
            WB[i] = guidedfilterbis(imList2[i], weightMapList[i], 23, 0.1)
            
           
            print('guidedfilter 2')
            WD[i] = guidedfilterbis(pic, weightMapList[i], 4, 10**(-7))
            
            

    else :
        for i in range(len(imList)):
            print('guidedfilter 1')
            pic = imList2[i][:,:]
            WB[i] = guidedfilter0(imList2[i], weightMapList[i], 45, 0.3)

           
            print('guidedfilter 2')
            WD[i] = guidedfilter0(pic, weightMapList[i], 7, 10**(-6))
    

    result = (fusion(baseLayerList, WB) + fusion(detailLayerList, WD))*255


    cv2.imwrite(fusionname, result) 
    Image.open(fusionname).show()



def testopencv(string, fusionname): #fonction execute mais avec la foncton guidedFilter d'opencv
    color = 1
    
    st =os.listdir(string)

    print(st[0]) #st[0] n'est pas un nom d'image donc onl'enlève de la liste à l'étape suivante

    if (st[0]=='.DS_Store'):
        imList = st[1:]
    else : 
        imList = st
    ### on prend souvent la précaution de copier la liste originale en utilisant [:] pour éviter de la modifier à chaque modification de la liste copiée

    l = len(imList)
    for i in range(l) :
        imList[i] = cv2.imread(string+'/'+imList[i]) #lecture des images du dossier

    if len(imList[0].shape)==2:
        color = 0 # séparation du cas où l'image est en couleur ou en niveau de gris
    

    imList1 = [imList[i]/255 for i in range(len(imList))]
    baseLayerList = imList[:]
    detailLayerList = []
    WB = imList[:]
    WD = imList[:] #création de listes de même taille et même type que la liste originale pour les modifier facilement

    print('baseLayer')
    for i in range(l): 
        baseLayerList[i] = baseLayer(imList1[i])
       

        detailLayerList.append(imList1[i]-baseLayerList[i])
        
    
    print('weightMap')
    weightMapList = weightMap(imList[:])


    imList2 = imList[:]
    weightMapList1 = weightMapList[:]

    if color ==1 :
        for i in range(len(imList)):
            print('guidedfilter 1')
            pic = imList2[i][:,:]
            WB[i] = guidedFilter(imList2[i], weightMapList[i], 5, 0.3*255)
           
            print('guidedfilter 2')
            WD[i] = guidedFilter(pic, weightMapList[i], 1, 10**(-6)*255)
            

    else :
        for i in range(len(imList)):
            print('guidedfilter 1')
            pic = imList2[i][:,:]
            WB[i] = guidedFilter(imList2[i], weightMapList[i], 45, 0.3)

           
            print('guidedfilter 2')
            WD[i] = guidedFilter(pic, weightMapList[i], 7, 10**(-6))
    

    result = (fusion(baseLayerList, WB) + fusion(detailLayerList, WD))


    cv2.imwrite(fusionname, result) 
    Image.open(fusionname).show()

















    