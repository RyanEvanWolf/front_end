import numpy as np
from cv_bridge import CvBridge
from front_end.utils import *
import cv2
cvb=CvBridge()

def drawStereoLandmarks(lmsg,rmsg,landmarks):
    left=cvb.imgmsg_to_cv2(lmsg)
    left=cv2.cvtColor(left,cv2.COLOR_GRAY2RGB)
    print(left.shape)
    right=cvb.imgmsg_to_cv2(rmsg)
    right=cv2.cvtColor(right,cv2.COLOR_GRAY2RGB)
    Epi=np.hstack((left,right))
    matches=[]
    for i in landmarks.out.matches:
        matches.append(ros2cv_dmatch(i))
    ##keeping here for future reference
    # for i in range(0,len(matches)):
    #     l=unpackKP(landmarks.out.leftFeatures)[i].pt
    #     l=(int(l[0]),int(l[1]))
    #     r=unpackKP(landmarks.out.rightFeatures)[i].pt
    #     r=(int(r[0])+left.shape[1],int(r[1]))###shape is [1] because its numpy array
    #     cv2.circle(Epi,l,2,(255,0,0),1)
    #     cv2.circle(Epi,r,2,(255,0,0),1)
    #     cv2.line(Epi,l,r,(255,0,0),1)
    # print(len(unpackKP(landmarks.out.leftFeatures)),
    #     len(unpackKP(landmarks.out.leftFeatures)),
    #     len(matches),
    #     len(landmarks.out.leftFeatures))
    Epi=cv2.drawMatches(left,unpackKP(landmarks.out.leftFeatures),
                    right,unpackKP(landmarks.out.rightFeatures),matches,None)
    return Epi
def drawMask(lkp,rkp,limg,rimg,mask,index=0):
    left=cv2.cvtColor(limg,cv2.COLOR_GRAY2RGB)
    right=cv2.cvtColor(rimg,cv2.COLOR_GRAY2RGB)
    #Epi=np.hstack((left,right))

    rightKPS=[]
    l=lkp[index]

    for i in range(0,len(rkp)):
        if(mask[index,i]==1):
            rightKPS.append(rkp[i])
    cv2.drawKeyPoints(left,(l),left)
    cv2.drawKeyPoints(right,rightKPS,right)

    Epi=np.hstack((left,right))  


    return Epi

def genStereoscopicImage(left,right):
    ###convert to colours
    print(left.shape,left.dtype)
    limg=np.zeros((left.shape[0],left.shape[1],3),dtype=np.uint8)
    limg[:,:,0]=0.8*left
    limg[:,:,1]=0.1*left
    limg[:,:,2]=0.1*left
    rimg=np.zeros((right.shape[0],right.shape[1],3),dtype=np.uint8)
    rimg[:,:,0]=0.1*right
    rimg[:,:,1]=0.1*right
    rimg[:,:,2]=0.8*right
    stereoscopic=cv2.addWeighted(limg,0.6,rimg,0.4,0)
    return stereoscopic

def drawFrameTracks(stereoImage,kpl,kpr,matches):
    copyImg=stereoImage
    for i in range(0,len(matches)):
        l=(int(kpl[matches[i].queryIdx].pt[0]),int(kpl[matches[i].queryIdx].pt[1]))
        r=(int(kpr[matches[i].trainIdx].pt[0]),int(kpr[matches[i].trainIdx].pt[1]))
        cv2.circle(copyImg,l,2,(0,255,0),1)
        cv2.circle(copyImg,r,2,(255,222,1),1)
        cv2.line(copyImg,l,r,(255,255,25),1)
    return copyImg
    # leftColour=cv2.cvtColor(self.left,cv2.COLOR_GRAY2RGB)
    # rightColour=cv2.cvtColor(self.right,cv2.COLOR_GRAY2RGB)
    # Epi=np.hstack((leftColour, rightColour))



        # def drawFeatures(self):
        # data=copy.deepcopy(self.getMatches())
        # leftColour=cv2.cvtColor(self.left,cv2.COLOR_GRAY2RGB)
        # rightColour=cv2.cvtColor(self.right,cv2.COLOR_GRAY2RGB)
        # Epi=np.hstack((leftColour, rightColour))

        # for featureIndex in range(0,len(data[0])):
        #     ##draw current features on left,right, and epipolar Image
        #     cv2.circle(leftColour,(int(data[CurrentIndex][featureIndex][LeftIndex].u),int(data[CurrentIndex][featureIndex][LeftIndex].v)),2,(0,255,0))
        #     cv2.circle(rightColour,(int(data[CurrentIndex][featureIndex][RightIndex].u),int(data[CurrentIndex][featureIndex][RightIndex].v)),2,(0,255,0))
        #     cv2.circle(Epi,(int(data[CurrentIndex][featureIndex][LeftIndex].u),int(round(data[CurrentIndex][featureIndex][LeftIndex].v))),2,(25,255,120))
        #     cv2.circle(Epi,(int(data[CurrentIndex][featureIndex][RightIndex].u)+leftColour.shape[1],int(round(data[CurrentIndex][featureIndex][RightIndex].v))),2,(25,255,120))
        #     ##draw previous features on left,right, and epipolar Image
        #     cv2.circle(leftColour,(int(data[PreviousIndex][featureIndex][LeftIndex].u),int(data[PreviousIndex][featureIndex][LeftIndex].v)),2,(255,0,0))
        #     cv2.circle(rightColour,(int(data[PreviousIndex][featureIndex][RightIndex].u),int(data[PreviousIndex][featureIndex][RightIndex].v)),2,(255,0,0))
        #     ##draw Tracked Lines
        #     cv2.line(leftColour,(int(data[CurrentIndex][featureIndex][LeftIndex].u),int(data[CurrentIndex][featureIndex][LeftIndex].v)),
        #                         (int(data[PreviousIndex][featureIndex][LeftIndex].u),int(data[PreviousIndex][featureIndex][LeftIndex].v)),(255,0,0),1)
        #     cv2.line(rightColour,(int(data[CurrentIndex][featureIndex][RightIndex].u),int(data[CurrentIndex][featureIndex][RightIndex].v)),
        #                         (int(data[PreviousIndex][featureIndex][RightIndex].u),int(data[PreviousIndex][featureIndex][RightIndex].v)),(255,0,0),1)

        #     #draw Epi Lines
        #     cv2.line(leftColour,(int(data[0][featureIndex][0].u),int(data[0][featureIndex][0].v)),
        #                         (int(data[1][featureIndex][0].u),int(data[1][featureIndex][0].v)),(255,0,0),1)

        #     #draw Epi Image
        #     cv2.line(Epi,(int(data[CurrentIndex][featureIndex][LeftIndex].u),int(round(data[CurrentIndex][featureIndex][LeftIndex].v))),
        #                         (int(data[CurrentIndex][featureIndex][RightIndex].u)+leftColour.shape[1],int(round(data[CurrentIndex][featureIndex][RightIndex].v))),(255,0,0),1)
        # return (leftColour,rightColour,Epi)
