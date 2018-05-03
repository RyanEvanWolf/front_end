#!/usr/bin/env python
#####ros related imports
import rospy 
import cv2
import time

from front_end.utils import *
from front_end.srv import *
from front_end.msg import frameDetection,ProcTime,kPoint

import itertools

from cv_bridge import CvBridge
import numpy as np

import pickle
import argparse

from statistics import mean,stdev
import matplotlib.pyplot as plt
import matplotlib.style as sty
cvb=CvBridge() 

##############
##FAST FEATURES
def getFAST_parameters():
    threshold=np.arange(1, 60, 3)
    dType=(cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    maxSuppression=(True,False)
    output={}
    output["threshold"]=threshold
    output["d_type"]=dType
    output["non_maximum_suppression"]=maxSuppression
    return output

def getFAST_combinations():
    output=[]
    params=getFAST_parameters()
    for t in params["threshold"]:
        for d in params["d_type"]:
            for n in params["non_maximum_suppression"]:
                singleSettings={}
                singleSettings["Name"]="FAST"
                singleSettings["Param"]=[]
                singleSettings["Param"].append(str(t))
                singleSettings["Param"].append(str(d))
                singleSettings["Param"].append(str(n))
                output.append(singleSettings)
    return output

def getFAST(params):
    detector=cv2.FastFeatureDetector_create()
    detector.setThreshold(int(params[0]))
    detector.setType(int(params[1]))
    detector.setNonmaxSuppression(bool(params[2]))
    return detector
###############
###BRIEF

def getBRIEF_parameters():
    size=[16,32,64]
    orientation=[1,0]
    output={}
    output["bytes"]=size
    output["use_orientation"]=orientation
    return output

def getBRIEF_combinations():
    output=[]
    params=getBRIEF_parameters()
    for b in params["bytes"]:
        for o in params["use_orientation"]:
            singleSettings={}
            singleSettings["Name"]="BRIEF"
            singleSettings["Param"]=[]
            singleSettings["Param"].append(str(b))
            singleSettings["Param"].append(str(o))
            singleSettings["NormType"]="NORM_HAMMING"
            output.append(singleSettings)
    return output

def getBRIEF(params):
    descr=cv2.xfeatures2d.BriefDescriptorExtractor_create(int(params[0]),bool(params[1]))
    #descr.setDescriptorSize(int(params[0]))
    return descr
###################
####SURF

def getSURF_parameters():
    threshold=np.arange(200,500,50)
    nOctave=np.arange(4,7,1)
    nOctaveLayers=np.arange(3,6,1)
    extended=(1,0)
    upright=(1,0)   
    output={}
    output["HessianThreshold"]=threshold
    output["nOctave"]=nOctave
    output["nOctaveLayers"]=nOctaveLayers
    output["Extended"]=extended
    output["Upright"]=upright
    return output

def getSURF_combinations():
    output=[]
    params=getSURF_parameters()
    for t in params["HessianThreshold"]:
        for n in params["nOctave"]:
            for on in params["nOctaveLayers"]:
                for e in params["Extended"]:
                    for u in params["Upright"]:
                        singleSettings={}
                        singleSettings["Name"]="SURF"
                        singleSettings["Param"]=[]
                        singleSettings["Param"].append(str(t))
                        singleSettings["Param"].append(str(n))
                        singleSettings["Param"].append(str(on))
                        singleSettings["Param"].append(str(e))
                        singleSettings["Param"].append(str(u))
                        singleSettings["NormType"]="NORM_L2"
                        output.append(singleSettings)
    return output

def getSURF(params):
    detector=cv2.xfeatures2d.SURF_create()
    detector.setHessianThreshold(float(params[0]))
    detector.setNOctaves(int(params[1]))
    detector.setNOctaveLayers(int(params[2]))
    detector.setExtended(int(params[3]))
    detector.setUpright(int(params[4]))
    return detector
    #         detectorRef.setHessianThreshold(float(parts[1]))
#         detectorRef.setNOctaves(int(parts[3]))
#         detectorRef.setNOctaveLayers(int(parts[5]))
#         detectorRef.setExtended(int(parts[7]))
#         detectorRef.setUpright(int(parts[9]))
#     for t in threshold:
#         for n in nOctave:
#             for nl in nOctaveLayers:
#                 for e in extended:
#                     for u in upright:
#                         msg="HessianThreshold,"+str(t)+",nOctave,"+str(n)+",nOctaveLayers,"+str(nl)+",Extended,"+str(e)+",Upright,"+str(u)
#                         detectorStrings.append(msg)
#     ##descriptor Settings
#     for e in extended:
#         for u in upright:
#             msg="SURF,HessianThreshold,"+str(threshold[0])+",nOctave,"+str(nOctave[0])+",nOctaveLayers,"+str(nOctaveLayers[1])+",Extended,"+str(e)+",Upright,"+str(u)
#             descriptorStrings.append(msg)
#     return output,detectorStrings,descriptorStrings

##############
####
def detectorLookUpTable():
    Table={}
    allSettings=(getFAST_combinations()+getSURF_combinations())
    for d in range(0,len(allSettings)):
        ID="Det"+str("%X" % d).zfill(10)
        Table[ID]=allSettings[d]
    return Table   

def descriptorLookUpTable():
    Table={}
    allSettings=(getBRIEF_combinations()
                + getSURF_combinations())
    for d in range(0,len(allSettings)):
        ID="Desc"+str("%X" % d).zfill(10)
        Table[ID]=allSettings[d]
    return Table

def getDetectorIDs(detectorName,lookupTable):
    idList=[]
    for i in lookupTable.key():
        if(lookupTable[i]["Name"]==detectorName):
            idList.append(i)
    return idList


def getDetector(Name,params):
    if(Name=="FAST"):
        return getFAST(params),True
    elif(Name=="SURF"):
        return getSURF(params),True
    else:
        return None,False

def getDescriptor(Name,params):
    if(Name=="SURF"):
        return getSURF(params),True
    if(Name=="BRIEF"):
        return getBRIEF(params),True
    else:
        return None,False

def getMatcher(normType):
    if(normType=="NORM_L2"):
        return cv2.BFMatcher(cv2.NORM_L2)
    elif(normType=="NORM_HAMMING"):
        return cv2.BFMatcher(cv2.NORM_HAMMING)

def assignIDs(listKP):
    for i in range(0,len(listKP)):
        listKP[i].class_id=i
#def updateDetector(Name,params,detectorRef)
# def updateDetector(name,csvString,detectorRef):
#     parts=csvString.split(",")
#     if(name=="FAST"):
#         detectorRef.setThreshold(int(parts[1]))
#         detectorRef.setType(int(parts[3]))
#         detectorRef.setNonmaxSuppression(bool(parts[5]))
#     if(name=="SURF"):
#         detectorRef.setHessianThreshold(float(parts[1]))
#         detectorRef.setNOctaves(int(parts[3]))
#         detectorRef.setNOctaveLayers(int(parts[5]))
#         detectorRef.setExtended(int(parts[7]))
#         detectorRef.setUpright(int(parts[9]))

# def updateDescriptor(csvString,detectorRef):
#     parts=csvString.split(",")

# def getDetector(name):
#     if(name=="FAST"):
#         return True,cv2.FastFeatureDetector_create()
#     elif(name=="SURF"):
#         return True,cv2.xfeatures2d.SURF_create()
#     else:
#         return False,None



def plotFeatures(pickledReference):
    nImages=len(pickledReference["data"])
    totalSettings=len(pickledReference["settings"])
    name=pickledReference["detectorName"]
    print("Total Images :",nImages,name,totalSettings)

#################
###returns the setting as a list
def getBestSettings(pickledReference):
    nImages=len(pickledReference["data"])
    totalSettings=len(pickledReference["settings"])
    name=pickledReference["detectorName"]
    Results={}
    graphLabels=[]
    Results["Maximum"]=[]
    Results["0.9Maximum"]=[]
    Results["0.8Maximum"]=[]
    Results["0.7Maximum"]=[]
    Results["0.6Maximum"]=[]
    Results["+Deviation"]=[]
    Results["Mean"]=[]
    Results["-Deviation"]=[]
    Results["Minimum"]=[]
    for imageIndex in pickledReference["data"]:
        leftNFeatures=[]
        ##convert each frames data into a list
        for frameIndex in imageIndex.outputFrames:
            leftNFeatures.append(frameIndex.nLeft)
        MaxInFrame=np.amax(leftNFeatures)
        MinInFrame=np.amin(leftNFeatures)
        MeanInFrame=mean(leftNFeatures)
        dev=stdev(leftNFeatures)
        dev_mean=MeanInFrame+dev
        IdealPerformanceTotals=[("Maximum",MaxInFrame),
                        ("0.9Maximum",0.9*MaxInFrame),
                        ("0.8Maximum",0.8*MaxInFrame),
                        ("0.7Maximum",0.7*MaxInFrame),
                        ("0.6Maximum",0.6*MaxInFrame),
                        ("+Deviation",MeanInFrame+dev),
                        ("Mean",MeanInFrame),
                        ("-Deviation",np.clip(MeanInFrame-dev,0,MaxInFrame)),
                        ("Minimum",MinInFrame)]
        for i in IdealPerformanceTotals:
            closestIndex=np.abs(np.array(leftNFeatures)-i[1]).argmin()
            Results[i[0]].append(pickledReference["settings"][closestIndex])
    return Results
############
###returns a list for plotting the feature statistics from a given set of detector settings
def getFeatureSummary(pickledReference):
    nImages=len(pickledReference["data"])
    totalSettings=len(pickledReference["settings"])
    name=pickledReference["detectorName"]
    Results={}
    graphLabels=[]
    Results["Maximum"]=[]
    Results["0.9Maximum"]=[]
    Results["0.8Maximum"]=[]
    Results["0.7Maximum"]=[]
    Results["0.6Maximum"]=[]
    Results["+Deviation"]=[]
    Results["Mean"]=[]
    Results["-Deviation"]=[]
    Results["Minimum"]=[]
    for imageIndex in pickledReference["data"]:
        leftNFeatures=[]
        ##convert each frames data into a list
        for frameIndex in imageIndex.outputFrames:
            leftNFeatures.append(frameIndex.nLeft)
        MaxInFrame=np.amax(leftNFeatures)
        MinInFrame=np.amin(leftNFeatures)
        MeanInFrame=mean(leftNFeatures)
        dev=stdev(leftNFeatures)
        dev_mean=MeanInFrame+dev
        IdealPerformanceTotals=[("Maximum",MaxInFrame),
                        ("0.9Maximum",0.9*MaxInFrame),
                        ("0.8Maximum",0.8*MaxInFrame),
                        ("0.7Maximum",0.7*MaxInFrame),
                        ("0.6Maximum",0.6*MaxInFrame),
                        ("+Deviation",MeanInFrame+dev),
                        ("Mean",MeanInFrame),
                        ("-Deviation",np.clip(MeanInFrame-dev,0,MaxInFrame)),
                        ("Minimum",MinInFrame)]
        for i in IdealPerformanceTotals:
            closestIndex=np.abs(np.array(leftNFeatures)-i[1]).argmin()
            Results[i[0]].append(leftNFeatures[closestIndex])
    return Results

