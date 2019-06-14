import cv2
import numpy as np
import os
import glob
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True, help="input image file")
args=vars(ap.parse_args())

trainImg = []
note=[]
max_match_count=[]
detector= cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

for filename in glob.glob('images/*.jpg'): #assuming gif
    im=cv2.imread(filename,0)
    trainImg.append(im)
    note.append(filename)
QueryImgBGR=cv2.imread(args["image"])
QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)

for num in range(len(note)):
    trainKP,trainDesc=detector.detectAndCompute(trainImg[num],None)
    
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)
    goodMatch=[]
    match_count=[]
    
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
            match_count.append(len(goodMatch))
    
    max_match_count.append(max(match_count))
print(max_match_count)
for num in range(len(note)):
    trainKP,trainDesc=detector.detectAndCompute(trainImg[num],None)
    h,w=trainImg[num].shape
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)
    goodMatch=[]
    
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)==max(max_match_count)):
        print(note[num],len(goodMatch))
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        cv2.putText(QueryImgBGR,str(note[num]),(10,40), cv2.FONT_HERSHEY_SIMPLEX, .9,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow('result',QueryImgBGR)
        print('match')
    else:
        pass
cv2.waitKey(0)

