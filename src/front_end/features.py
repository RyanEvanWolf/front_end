#!/usr/bin/env python
#####ros related imports
import rospy 
import cv2
import time
import copy
from bumblebee.stereo import *
from bumblebee.baseTypes import *

from front_end.utils import *
from front_end.srv import *
from front_end.msg import frameDetection,ProcTime,kPoint,cvMatch,stereoFeatures
from Queue import Queue
import itertools

from cv_bridge import CvBridge
import numpy as np

import pickle
import argparse

from statistics import mean,stdev
import matplotlib.pyplot as plt
import matplotlib.style as sty
cvb=CvBridge() 


from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from front_end.srv import controlDetection,controlDetectionResponse

import time

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

###################################################################
####live features

class gridDetector:
    def __init__(self):
        self.row=2
        self.col=3
        self.updateSetPoint(3000)
        self.Thresholds=10*np.ones((self.row,self.col),dtype=np.uint8)
        self.detector=cv2.FastFeatureDetector_create()
        self.detector.setType(cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)
        self.detector.setNonmaxSuppression(True)
        self.x=None
        self.y=None
        self.w=None
        self.h=None
        self.winSize=(5,5)
        self.zeroZone=(-1,-1)
    def updateSetPoint(self,setPoint):
        self.bucketSetpoint=int(setPoint/float(self.row*self.col))
        self.setPoint=setPoint
    def updateThreshold(self,threshold):
        self.Thresholds=threshold*np.ones((self.row,self.col),dtype=np.uint8)
    def detect(self,rectifiedImg,update=True):
        roiIMG=rectifiedImg[self.y:self.h+1,self.x:self.w+1]
        imgWidth=int(self.w/self.col)                        
        imgHeight=int(self.h/self.row)
        Detections=[]
        bottomDetections=[]
        for row in range(self.row):
            for col in range(self.col):
                xOffset=col*imgWidth
                yOffset=row*imgHeight
                miniIMG=roiIMG[yOffset:yOffset+imgHeight,xOffset:xOffset+imgWidth]
                self.detector.setThreshold(self.Thresholds[row,col])
                detections=self.detector.detect(miniIMG)
                
                for d in detections:
                    d.pt=(d.pt[0]+xOffset+self.x,d.pt[1]+yOffset+self.y)
                Detections=Detections+detections
                if(row==1):
                    error=len(detections)-2*self.bucketSetpoint
                    hysteresis=0.2*2*self.bucketSetpoint
                else:
                    error=len(detections)-0.5*self.bucketSetpoint
                    hysteresis=0.2*0.5*self.bucketSetpoint
                if(abs(error)>hysteresis):
                    if(error>0):
                        self.Thresholds[row,col]=np.clip(self.Thresholds[row,col]+1 ,6,80)   #3
                    else:
                        self.Thresholds[row,col]=np.clip(self.Thresholds[row,col]-1 ,6,80) #3
        for k in Detections:
            refinement=np.float32([k.pt])
            cv2.cornerSubPix(rectifiedImg,refinement, self.winSize, self.zeroZone, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,40, 0.001))
            k.pt=(refinement[0,0],refinement[0,1])
        return Detections

class stereoDetector:
    def __init__(self,fullDebug=False):
        self.fullDebug=fullDebug
        self.kSettings=getCameraSettingsFromServer(cameraType="subROI")
        self.topic=["Dataset/left","Dataset/right"]
        self.q=[Queue(),Queue()]
        self.cvb=CvBridge()
        self.sub=[rospy.Subscriber(self.topic[0],Image,self.updateFeature,"l"),rospy.Subscriber(self.topic[1],Image,self.updateFeature,"r")]
        self.outPub=rospy.Publisher("stereo/Features",stereoFeatures,queue_size=10)
        self.lDetector,self.rDetector=gridDetector(),gridDetector()

        roi=ROIfrmMsg(self.kSettings["lInfo"].roi)
        x,y,w,h=roi[0],roi[1],roi[2],roi[3]
        self.lDetector.x=roi[0]
        self.lDetector.y=roi[1]
        self.lDetector.w=roi[2]
        self.lDetector.h=roi[3]

        self.rDetector.x=roi[0]
        self.rDetector.y=roi[1]
        self.rDetector.w=roi[2]
        self.rDetector.h=roi[3]

        self.centreROI=y+int(h/2.0)

        self.descr=cv2.xfeatures2d.BriefDescriptorExtractor_create(16,False)

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.debugResults=([rospy.Publisher("stereo/debug/matches",Float32,queue_size=1),
                            rospy.Publisher("stereo/debug/detection",Float32,queue_size=1),
                            rospy.Publisher("stereo/time/detection",Float32,queue_size=1),
                            rospy.Publisher("stereo/time/matches",Float32,queue_size=1)])
        self.controlDetection=rospy.Service("stereo/control/detection",controlDetection,self.resetDetection)                  
        if(fullDebug):
            self.debugResults.append(rospy.Publisher("stereo/image/matches",Image,queue_size=1))
            self.debugResults.append(rospy.Publisher("stereo/image/detection",Image,queue_size=1))
    def resetDetection(self,req):
        
        self.lDetector.updateSetPoint(req.setPoint)
        self.lDetector.updateThreshold(req.threshold)
        
        self.rDetector.updateSetPoint(req.setPoint)
        self.rDetector.updateThreshold(req.threshold)
        res=controlDetectionResponse()
        res.newSetPoint=self.lDetector.setPoint
        return res
    def updateFeature(self,data,arg):
        print("img",time.time())
        if(arg=="l"):
            self.q[0].put(self.cvb.imgmsg_to_cv2(data))
        else:
            self.q[1].put(self.cvb.imgmsg_to_cv2(data))
    def update(self):
        if(self.q[0].qsize()>0 and self.q[1].qsize()>0):
            lImg=self.q[0].get()
            rImg=self.q[1].get()
            ###########
            ##apply ROI
            self.compute(lImg,rImg)
    def compute(self,limg,rimg):
        ####
        #ROi
        debugData=Float32()
        
        a=time.time()
        l=[]
        detectTime=time.time()
        lKP=self.lDetector.detect(limg)
        rKP=self.rDetector.detect(rimg)
        detectTime=time.time()-detectTime
        debugData.data=detectTime
        self.debugResults[2].publish(debugData)

        debugData.data=len(lKP)
        self.debugResults[1].publish(debugData)

        computeTime=time.time()
        lKP,lDesc=self.descr.compute(limg,lKP)
        rKP,rDesc=self.descr.compute(rimg,rKP)
        # Match descriptors.
        matches =self.bf.match(lDesc,rDesc)
        goodMatches=[]
        goodLdesc=[]
        goodRdesc=[]
        goodMatches=stereoFeatures()
        inlierTopMatches=[]
        inlierBottomMatches=[]
        for m in matches:
            vDist=lKP[m.queryIdx].pt[1]-rKP[m.trainIdx].pt[1]
            if(abs(vDist)<=0.7):
               goodMatches.leftFeatures.append(cv2ros_KP(lKP[m.queryIdx]))
               goodMatches.rightFeatures.append(cv2ros_KP(rKP[m.trainIdx]))
               goodMatches.matchScore.append(m.distance)
               inlierTopMatches.append(m)
               #print(lDesc[m.queryIdx,:].shape)
               goodLdesc.append(lDesc[m.queryIdx,:])#.reshape(1,16))
               goodRdesc.append(rDesc[m.trainIdx,:])#.reshape(1,16))
        computeTime= time.time()-computeTime

        debugData.data=len(goodMatches.leftFeatures)
        self.debugResults[0].publish(debugData)
        debugData.data=computeTime
        self.debugResults[3].publish(debugData)
        


        # #####
        # ###pack the descriptors into the message
        if(self.fullDebug):
            abcd=cv2.cvtColor(limg,cv2.COLOR_GRAY2RGB)
            im_with_keypoints = cv2.drawKeypoints(limg,lKP, np.array([]), (255,0,20), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)   

            self.debugResults[5].publish(self.cvb.cv2_to_imgmsg(im_with_keypoints))
            img3=np.hstack((copy.deepcopy(limg),copy.deepcopy(rimg)))
            #cv2.drawKeypoints(limg,lKP,img3,(150,200,0))
            img3=cv2.drawMatches(limg,lKP,rimg,rKP,inlierTopMatches,img3,(0,255,0) ,flags=2)
            self.debugResults[4].publish(self.cvb.cv2_to_imgmsg(img3))
        
        
        descriptorUnitType=goodLdesc[0].dtype
        descriptorLength=goodLdesc[0].shape[0]
        print(descriptorUnitType,descriptorLength)       
        
        
        
        packedL=np.zeros((len(goodLdesc),descriptorLength),dtype=descriptorUnitType)
        packedR=np.zeros((len(goodRdesc),descriptorLength),dtype=descriptorUnitType)
        for i in range(0,len(goodLdesc)):
            packedL[i,:]=copy.deepcopy(goodLdesc[i])
            packedR[i,:]=copy.deepcopy(goodRdesc[i])
        goodMatches.leftDescr=self.cvb.cv2_to_imgmsg(packedL)
        goodMatches.rightDescr=self.cvb.cv2_to_imgmsg(packedR)

        self.outPub.publish(goodMatches)
        print("published @ ",time.time())
















        # img3=np.hstack((copy.deepcopy(lROI),copy.deepcopy(rROI)))
        # img3=cv2.drawMatches(lROI,lKP,rROI,rKP,goodMatches,img3, flags=2)

        # cv2.imshow("a",img3)
        # cv2.waitKey(150)







       # plt.imshow(img3),plt.show()
#         thresh=
# #cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
#         for i in range(maxIter):
#             detector.setThreshold(2+i*5)
#             l=detector.detect(lROI)
#             r=detector.detect(rROI)
#             coarse.append(abs(setPoint-len(l)))
#         best=min(coarse)
#         ind=coarse.index(best)
#         print(coarse,best)
#         drawn=copy.deepcopy(lROI)
#         drawn=cv2.drawKeypoints(lROI,l,drawn)

#         k=cv2.Canny(lROI,20,120)
#         cv2.imshow("a",drawn)
#         cv2.imshow("b",k)
#         #cv2.imshow("b",limg)

#         cv2.waitKey(100)

#         print(time.time())
