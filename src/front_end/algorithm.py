import numpy as np
from front_end.utils import *
from front_end.features import *



EPI_THRESHOLD=2.0
LOWE_THRESHOLD=0.8

def getEpiPolarMatches(leftKP,rightKP):
    ##build Distance Table
    mask=np.zeros((len(leftKP),len(rightKP)),dtype=np.uint8)
    distances=np.zeros((len(leftKP),len(rightKP)),dtype=np.float64)
    for row in range(0,distances.shape[0]):
        for col in range(0,distances.shape[1]):
            distances[row,col]=abs(leftKP[row].pt[0]-rightKP[col].pt[0])
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
def algorithm_one(stereoFeatIn):
    ###Filter
    lkp=unpackKP(stereoFeatIn.leftFeatures)
    rkp=unpackKP(stereoFeatIn.rightFeatures)
    ###KNN Match
    ###
    return True