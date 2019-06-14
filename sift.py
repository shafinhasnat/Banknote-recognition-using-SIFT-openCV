import cv2
import numpy as np
import os
import glob

trainImg = []
note=[]
MIN_MATCH_COUNT=30

detector= cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

for filename in glob.glob('images/*.jpg'): #assuming gif
    im=cv2.imread(filename,0)
    trainImg.append(im)
    note.append(filename)


print(note)
cam=cv2.VideoCapture('http://192.168.0.100:8080/video')
while True:
    ret, QueryImgBGR=cam.read()
    QueryImgBGR=cv2.resize(QueryImgBGR,(640,360))
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    for num in range(6):
        trainKP,trainDesc=detector.detectAndCompute(trainImg[num],None)
        h,w=trainImg[num].shape
        matches=flann.knnMatch(queryDesc,trainDesc,k=2)

        goodMatch=[]
        for m,n in matches:
            if(m.distance<0.75*n.distance):
                goodMatch.append(m)
        if(len(goodMatch)>MIN_MATCH_COUNT):
            print(note[num])
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
        else:
    #        for i in range(2q):
    #            print(i)
    #            n=i
            pass
        cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
