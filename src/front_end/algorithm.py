import numpy as np
from front_end.utils import *
from front_end.features import *
from front_end.srv import *
from cv_bridge import CvBridge
from front_end.msg import ProcTime,kPoint,stereoLandmarks

EPI_THRESHOLD=2.0
LOWE_THRESHOLD=0.8

###################
###Algorithm One

def getEpiPolarMatches(leftKP,rightKP):
    ##build Distance Table
    mask=np.zeros((len(leftKP),len(rightKP)),dtype=np.uint8)
    distances=np.zeros((len(leftKP),len(rightKP)),dtype=np.float64)
    for row in range(0,distances.shape[0]):
        for col in range(0,distances.shape[1]):
            distances[row,col]=abs(leftKP[row].pt[1]-rightKP[col].pt[1])
    for row in range(0,distances.shape[0]):
        for col in range(0,distances.shape[1]):
            if(distances[row,col]<=EPI_THRESHOLD):
                mask[row,col]=1
    return mask,distances

def loweFilterPotential(matches):
    goodMatches=[]
    for i in matches:
        if(len(i)==1):
            goodMatches.append(i[0])
        elif(len(i)>1):
            if(i[0].distance<LOWE_THRESHOLD*i[1].distance):
                goodMatches.append(i[0])
    return goodMatches

def getPotentialMatches(leftDescr,rightDescr,mask,norm):
    matcher=getMatcher(norm)
    ####unpack descriptors
    ###left Descriptors
    ans=matcher.knnMatch(leftDescr,rightDescr,2,mask)
    return ans
def algorithm_one(stereoFrame):
    cvb=CvBridge()
    descTable=descriptorLookUpTable()

    print("received ",stereoFrame.detID,stereoFrame.descrID)
    ###unpack the keypoints into cv lists
    lkp=unpackKP(stereoFrame.leftFeatures)
    assignIDs(lkp)
    rkp=unpackKP(stereoFrame.rightFeatures)
    assignIDs(rkp)
    ld=cvb.imgmsg_to_cv2(stereoFrame.leftDescr)
    rd=cvb.imgmsg_to_cv2(stereoFrame.rightDescr)
    ###filter by epipolar matches

    print("Matching")
    epiTime=ProcTime()
    epiTime.label="Epipolar Filter"
    MatchTime=ProcTime()
    MatchTime.label="KNN Match"
    loweTime=ProcTime()
    loweTime.label="lowe Ratio"

    startTime=time.time()
    mask,dist=getEpiPolarMatches(lkp,rkp)
    epiTime.seconds=time.time()-startTime


    startTime=time.time()
    initialMatches=getPotentialMatches(ld,
                          rd,
                          mask,descTable[stereoFrame.descrID]["NormType"])
    MatchTime.seconds=time.time()-startTime

    startTime=time.time()
    finalMatches=loweFilterPotential(initialMatches)
    loweTime.seconds=time.time()-startTime

    reply=stereoMatchingResponse()
    #####pack into a frame
    msg=stereoLandmarks()
    msg.detID=stereoFrame.detID
    msg.descrID=stereoFrame.descrID
    newLdesc=np.zeros((len(finalMatches),ld.shape[1]),dtype=ld.dtype)
    newRdesc=np.zeros((len(finalMatches),ld.shape[1]),dtype=ld.dtype)
    for index in range(0,len(finalMatches)):
        ###pack left 
        msg.leftFeatures.append(cv2ros_KP(lkp[finalMatches[index].queryIdx]))
        newLdesc[index,:]=ld[finalMatches[index].queryIdx,:]
        ###pack right
        msg.rightFeatures.append(cv2ros_KP(rkp[finalMatches[index].trainIdx]))
        newRdesc[index,:]=rd[finalMatches[index].trainIdx,:]
        ###pack match
        match=cv2.DMatch()
        match.distance=finalMatches[index].distance
        match.queryIdx=index
        match.trainIdx=index
        match.imgIdx=0
        msg.matches.append(cv2ros_dmatch(match))
    msg.leftDescr=cvb.cv2_to_imgmsg(newLdesc)
    msg.rightDescr=cvb.cv2_to_imgmsg(newRdesc)
    print(len(msg.leftFeatures),len(msg.rightFeatures))
    reply.out=msg
    reply.out.proc.append(epiTime)
    reply.out.proc.append(MatchTime)
    reply.out.proc.append(loweTime)
    return reply

###################
###Algorithm Two