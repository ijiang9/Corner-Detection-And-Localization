#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:10:57 2020

@author: yinjiang
"""

import cv2
import numpy as np

"""
Harris corner detector with localiztion
Returns corner window center(red point) and location
of the corner(green point)
"""
def cornerH(image, s, winSize, k, threshold):
    threshold = int(threshold)
    s = int(s)
    winSize = int(winSize)
    k = float(k)
    if winSize%2==0:
        winSize+=1
    halfWin = int(winSize/2)
    
    #convert to grayscale
    img_p = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    
    #smoothing
    img_p = cv2.GaussianBlur(img_p,(s,s),0)
    
    #gradient
    dx = cv2.Sobel(img_p,cv2.CV_64F,1,0,ksize=3)
    dy = cv2.Sobel(img_p,cv2.CV_64F,0,1,ksize=3)
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2
    
    #list of corners
    corners = []
    #list of corner locations
    kp = []
    #for each point in the image, perform corner detection
    for i in range(halfWin, img_p.shape[0]-halfWin):
        for j in range(halfWin, img_p.shape[1]-halfWin):
            wIxx = Ixx[i-halfWin:i+halfWin+1,j-halfWin:j+halfWin+1]
            wIxy = Ixy[i-halfWin:i+halfWin+1,j-halfWin:j+halfWin+1]
            wIyy = Iyy[i-halfWin:i+halfWin+1,j-halfWin:j+halfWin+1]
            Sxx = wIxx.sum()
            Sxy = wIxy.sum()
            Syy = wIyy.sum()
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            #harris score
            HC = det - k*(trace**2)
            #thresholding
            if HC > threshold:
                #building vector of gradient vectors
                D = [dx[i-halfWin:i+halfWin+1,j-halfWin:j+halfWin+1],dy[i-halfWin:i+halfWin+1,j-halfWin:j+halfWin+1]]
                D = np.asarray(D)
                D = np.reshape(D,(2,winSize**2,-1))
                D = D.T
                D = D[0]
                #Correlation matrix
                C = np.dot(D.T,D)
                #find location p*
                p = np.zeros((2,1))
                for ii in range(i-halfWin,i+halfWin+1):
                    for jj in range(j-halfWin,j+halfWin+1):
                        #gradient vector
                        g = np.reshape(np.asarray([dx[ii,jj],dy[ii,jj]]),(1,2))
                        #xi
                        xi = np.reshape(np.asarray([ii,jj]),(1,2))
                        #sum gradient vector * xi
                        p = p + np.dot(np.dot(g.T,g),xi.T)
                #p* = inv(C) * sum(g*g.T*xi)
                idx = np.dot(np.linalg.inv(C),p)
                corners.append([i,j])
                kp.append([int(idx[0]),int(idx[1])])
    
    return corners, kp
"""draw rectangle around center of the window (blue)
   set center of the window (red)
   set location of the corner (green)
"""
def drawRectangle(corners,kp,img):
    img_p = img.copy()
    color = (255, 0, 0) 
    #rectangle and center
    for index in corners:
        start = (index[1]+10,index[0]+10)
        end = (index[1]-10, index[0]-10)
        img_p.itemset((index[0],index[1],0),0)
        img_p.itemset((index[0],index[1],1),0)
        img_p.itemset((index[0],index[1],2),255)
        cv2.rectangle(img_p, start, end, color, 1)
    #location
    for k in kp:
        img_p.itemset((k[0],k[1],0),0)
        img_p.itemset((k[0],k[1],1),255)
        img_p.itemset((k[0],k[1],2),0)

    return img_p
"""
Find all matching key points using SIFT
Keep the matched key point if it is a corner point found in corner detection
github ref: https://github.com/chen910/cs512/blob/master/AS3/src/AS3.py
"""    
def featureVector(image1, image2,ckp1,ckp2):
	# Initiate SIFT detector
    orb = cv2.ORB_create()
	# find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(image1,None) # returns keypoints and descriptors
    kp2, des2 = orb.detectAndCompute(image2,None)
	# create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# Match descriptors.
    matches = bf.match(des1,des2)
	# Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    kp1List = []
    kp2List = []
    for m in matches:
        (x1, y1) = kp1[m.queryIdx].pt
        (x2, y2) = kp2[m.trainIdx].pt
        #feature vector of corner point in corner detection
        if [int(y1), int(x1)] in ckp1 and [int(y2), int(x2)] in ckp2:
            kp1List.append((x1, y1))
            kp2List.append((x2, y2))
    for i in range(0, len(kp1List)):
        point1 = kp1List[i]
        point2 = kp2List[i]
        cv2.putText(image1, str(i), (int(point1[0]), int(point1[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.putText(image2, str(i), (int(point2[0]), int(point2[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    rst = np.concatenate((image1, image2), axis=1)
    return rst

def help():
    print("'c': Find corners and better location")
    print("'f': Compute feature vector for corner and find matches")
    print("'q': Exit")
     
if __name__ == '__main__':
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    corners1 = 0
    corners2 = 0
    ckp1 = 0
    ckp2 = 0
    while True:
        key = input()
        if key == 'c':
            #3,3,0.04,100000000
            s = input("the varience of Gussian:")
            winSize = input("windowSize :")
            k = input("the weight of the harris conner detector:")
            threshold = input("threshold:")
            print("Computing...")
            corners1, ckp1 = cornerH(img1,s,winSize,k,threshold)
            corners2, ckp2 = cornerH(img2,s,winSize,k,threshold)
            print("Draw...")
            img_p1 = drawRectangle(corners1, ckp1, img1)
            img_p2 = drawRectangle(corners2, ckp2, img2)
            combine = np.concatenate((img_p1, img_p2), axis=1)
            combine = cv2.resize(combine,(int(combine.shape[1]/2),int(combine.shape[0]/2)))
            cv2.imshow("corners", combine)
            while True:
                if cv2.waitKey(10)==27:
                    cv2.destroyWindow("corners")
                    break
#            
        elif key == 'f':
            if corners1 == 0:
                print("Please find corners first!")
                continue
            combine = featureVector(img1, img2, ckp1,ckp2)
            combine = cv2.resize(combine,(int(combine.shape[1]/2),int(combine.shape[0]/2)))
            cv2.imshow("feature", combine)
            while True:
                if cv2.waitKey(10)==27:
                    cv2.destroyWindow("feature")
                    break
        elif key == 'h':
            help()
        elif key == 'q':
            break
        else:
            print("Not Implemented")
        
                
        
