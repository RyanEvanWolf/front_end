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
    threshold=np.arange(1, 60, 1)
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
    threshold=np.arange(10,550,10)
    nOctave=np.arange(2,6,2)
    nOctaveLayers=np.arange(2,6,1)
    extended=(1,0)
    upright=(1,0)   
    output={}
    output["HessianThreshold"]=threshold
    output["nOctave"]=nOctave
    output["nOctaveLayers"]=nOctaveLayers
    output["Extended"]=extended
    output["Upright"]=upright
    return output

def getSURF_DetectorCombinations():
    output=[]
    params=getSURF_parameters()
    for t in params["HessianThreshold"]:
        for n in params["nOctave"]:
            for on in params["nOctaveLayers"]:
                singleSettings={}
                singleSettings["Name"]="SURF"
                singleSettings["Param"]=[]
                singleSettings["Param"].append(str(t))
                singleSettings["Param"].append(str(n))
                singleSettings["Param"].append(str(on))
                singleSettings["Param"].append(str(params["Extended"][0]))###not used in detection
                singleSettings["Param"].append(str(params["Upright"][0]))###not used in detection
                singleSettings["NormType"]="NORM_L2"
                output.append(singleSettings)
    return output    

def getSURF_DescriptorCombinations():
    output=[]
    params=getSURF_parameters()
    for e in params["Extended"]:
        for u in params["Upright"]:
            singleSettings={}
            singleSettings["Name"]="SURF"
            singleSettings["Param"]=[]
            singleSettings["Param"].append(str(params["HessianThreshold"][0]))
            singleSettings["Param"].append(str(params["nOctave"][0]))
            singleSettings["Param"].append(str(params["nOctaveLayers"][0]))
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

######################
###BRISK
#####################


def getBRISK_parameters():
    threshold=np.arange(4,70,3)
    nOctave=np.arange(2,6,2)
    patternScale=np.linspace(0.8,2.5,10)
    output={}
    output["Threshold"]=threshold
    output["nOctave"]=nOctave
    output["patternScale"]=patternScale
    return output

def getBRISK_combinations():
    output=[]
    params=getBRISK_parameters()
    for t in params["Threshold"]:
        for n in params["nOctave"]:
            for on in params["patternScale"]:
                singleSettings={}
                singleSettings["Name"]="BRISK"
                singleSettings["Param"]=[]
                singleSettings["Param"].append(str(t))
                singleSettings["Param"].append(str(n))
                singleSettings["Param"].append(str(on))
                singleSettings["NormType"]="NORM_HAMMING"
                output.append(singleSettings)
    return output

def getBRISK(params):
    detector=cv2.BRISK_create(int(params[0]),
                            int(params[1]),
                            float(params[2]))
    return detector

####################
###AKAZE
####################
def getAKAZE_parameters():
    descriptorType=(cv2.AKAZE_DESCRIPTOR_KAZE,cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT,
                    cv2.AKAZE_DESCRIPTOR_MLDB,cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT)
    descriptorSize=(64,256,486)
    thresh=np.linspace(0.0001,0.02,24)
    nOctave=np.arange(2,6,2)
    nOctaveLayers=np.arange(2,6,2)
    diffuse=(cv2.KAZE_DIFF_WEICKERT,cv2.KAZE_DIFF_CHARBONNIER,
            cv2.KAZE_DIFF_PM_G1,cv2.KAZE_DIFF_PM_G2)
    output={}
    output["Threshold"]=thresh
    output["nOctave"]=nOctave
    output["nOctaveLayers"]=nOctaveLayers
    output["descriptorSize"]=descriptorSize
    output["descriptorType"]=descriptorType
    output["diffusivity"]=diffuse
    return output

def getAKAZE_DetectorCombinations():
    output=[]
    params=getAKAZE_parameters()
    for t in params["Threshold"]:
        for n in params["nOctave"]:
            for nl in params["nOctaveLayers"]:
                for diff in params["diffusivity"]:
                    singleSettings={}
                    singleSettings["Name"]="AKAZE"
                    singleSettings["Param"]=[]
                    singleSettings["Param"].append(str(params["descriptorSize"][0]))
                    singleSettings["Param"].append(str(params["descriptorType"][0]))
                    singleSettings["Param"].append(str(t))
                    singleSettings["Param"].append(str(n))
                    singleSettings["Param"].append(str(nl))
                    
                    singleSettings["Param"].append(str(diff))
                    singleSettings["NormType"]="NORM_HAMMING"
                    output.append(singleSettings)
    return output   

def getAKAZE_DescriptorCombinations():
    output=[]
    params=getAKAZE_parameters()
    for ds in params["descriptorSize"]:
        for dt in params["descriptorType"]:
            singleSettings={}
            singleSettings["Name"]="AKAZE"
            singleSettings["Param"]=[]
            singleSettings["Param"].append(str(dt))
            singleSettings["Param"].append(str(ds))
            singleSettings["Param"].append(str(params["Threshold"][0]))
            singleSettings["Param"].append(str(params["nOctave"][0]))
            singleSettings["Param"].append(str(params["nOctaveLayers"][0]))
            
            singleSettings["Param"].append(str(params["diffusivity"][0]))
            singleSettings["NormType"]="NORM_HAMMING"
            output.append(singleSettings)
    return output   

# def getAKAZE_combinations():
#     output=[]
#     params=getAKAZE_parameters()
#     for t in params["Threshold"]:
#         for n in params["nOctave"]:
#             for nl in params["nOctaveLayers"]:
#                 for ds in params["descriptorSize"]:
#                     for dt in params["descriptorType"]:
#                         for diff in params["diffusivity"]:
#                             singleSettings={}
#                             singleSettings["Name"]="AKAZE"
#                             singleSettings["Param"]=[]
#                             singleSettings["Param"].append(str(dt))
#                             singleSettings["Param"].append(str(ds))
#                             singleSettings["Param"].append(str(t))
#                             singleSettings["Param"].append(str(n))
#                             singleSettings["Param"].append(str(nl))
                            
#                             singleSettings["Param"].append(str(diff))
#                             singleSettings["NormType"]="NORM_HAMMING"
#                             output.append(singleSettings)
#     return output

def getAKAZE(params):
    detector=cv2.AKAZE_create(int(params[0]),
                               int(params[1]),
                               3,
                               float(params[2]),
                               int(params[3]),
                               int(params[4]),
                               int(params[5]))

    return detector
####################
###ORB
####################
def getORB_parameters():
    scaleFactor=np.linspace(1.1,2.0,8)
    nlevels=np.arange(2,6,2)
    Edgethreshold=np.arange(5,50,10)
    wta=(3,4)
    score=[cv2.ORB_FAST_SCORE]#,cv2.ORB_HARRIS_SCORE]
    patchSize=np.arange(10,70,20)
    threshold=np.arange(1,50,6)
    output={}
    output["scaleFactor"]=scaleFactor
    output["edgeThreshold"]=Edgethreshold
    output["nLevels"]=nlevels
    output["wta"]=wta
    output["scoreType"]=score
    output["patchSize"]=patchSize
    output["fastThreshold"]=threshold
    return output

def getORB_DetectorCombinations():
    output=[]
    params=getORB_parameters()
    for s in params["scaleFactor"]:
        for n in params["nLevels"]:
            for e in params["edgeThreshold"]:
                #for w in params["wta"]:
                    #for t in params["scoreType"]:
                    #    for p in params["patchSize"]:
                for th in params["fastThreshold"]:
                    singleSettings={}
                    singleSettings["Name"]="ORB"
                    singleSettings["Param"]=[]
                    singleSettings["Param"].append(str(s))
                    singleSettings["Param"].append(str(n))
                    singleSettings["Param"].append(str(e))
                    singleSettings["Param"].append(str(params["wta"][0]))
                    singleSettings["Param"].append(str(params["scoreType"][0]))
                    singleSettings["Param"].append(str(e))
                    singleSettings["Param"].append(str(th))
                    singleSettings["NormType"]="NORM_HAMMING"
                    output.append(singleSettings)
    return output

def getORB_DescriptorCombinations():
    output=[]
    params=getORB_parameters()
    for w in params["wta"]:
        for p in params["patchSize"]:
            singleSettings={}
            singleSettings["Name"]="ORB"
            singleSettings["Param"]=[]
            singleSettings["Param"].append(str(params["scaleFactor"][0]))

            singleSettings["Param"].append(str(params["nLevels"][0]))
            singleSettings["Param"].append(str(params["edgeThreshold"][0]))
            singleSettings["Param"].append(str(w))
            singleSettings["Param"].append(str(params["scoreType"][0]))
            singleSettings["Param"].append(str(params["edgeThreshold"][0]))
            singleSettings["Param"].append(str(params["fastThreshold"][0]))
            singleSettings["NormType"]="NORM_HAMMING"
            output.append(singleSettings)
    return output    

# def getORB_combinations():
#     output=[]
#     params=getORB_parameters()
#     for s in params["scaleFactor"]:
#         for n in params["nLevels"]:
#             for e in params["edgeThreshold"]:
#                 for w in params["wta"]:
#                     for t in params["scoreType"]:
#                         for p in params["patchSize"]:
#                             for th in params["fastThreshold"]:
#                                 singleSettings={}
#                                 singleSettings["Name"]="ORB"
#                                 singleSettings["Param"]=[]
#                                 singleSettings["Param"].append(str(s))
#                                 singleSettings["Param"].append(str(n))
#                                 singleSettings["Param"].append(str(e))
#                                 singleSettings["Param"].append(str(w))
#                                 singleSettings["Param"].append(str(t))
#                                 singleSettings["Param"].append(str(p))
#                                 singleSettings["Param"].append(str(th))
#                                 singleSettings["NormType"]="NORM_HAMMING"
#                                 output.append(singleSettings)
#     return output

def getORB(params):
    detector=cv2.ORB_create(25000,float(params[0]),
                    int(params[1]),
                    int(params[2]),
                    0,
                    int(params[3]),
                    int(params[4]),
                    int(params[5]),
                    int(params[6]))
    return detector
########################################################################
##############
####Lookup Tables and ID assignment
############
########################################################################
def getDetectorIDs(detectorName):
    table=detectorLookUpTable()
    results=[]
    for i in table.keys():
        if(table[i]["Name"]==detectorName):
            results.append(i)
    return results

def getDescriptorIDs(descriptorName):
    table=descriptorLookUpTable()
    results=[]
    for i in table.keys():
        if(table[i]["Name"]==descriptorName):
            results.append(i)
    return results
def detectorLookUpTable():
    Table={}
    allSettings=(getORB_DetectorCombinations()+
                 getBRISK_combinations()+
                getFAST_combinations()+
                getSURF_DetectorCombinations()+
                getBRISK_combinations()+
                getAKAZE_DetectorCombinations())
    #getAKAZE_combinations()#getBRISK_combinations()#(getFAST_combinations()+
               # getSURF_combinations()+#
               # getBRISK_combinations())
    for d in range(0,len(allSettings)):
        ID="Det"+str("%X" % d).zfill(10)
        Table[ID]=allSettings[d]
    return Table   

def descriptorLookUpTable():
    Table={}
    allSettings=getSURF_DescriptorCombinations()#(getBRIEF_combinations()
                #+ getSURF_combinations())
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
    elif(Name=="BRISK"):
        return getBRISK(params),True
    elif(Name=="AKAZE"):
        return getAKAZE(params),True
    elif(Name=="ORB"):
        return getORB(params),True
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

