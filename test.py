import cv2
import numpy as np
img = cv2.imread('C:/Users/ASUS-PC/Desktop/FeatureMatch/images/2_coin_b.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,None)

cv2.imshow('sift_keypoints',img)
