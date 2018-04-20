#!/usr/bin/env python
#####ros related imports
import rospy 
import cv2
import time

from front_end.utils import *
from front_end.srv import *
from front_end.msg import frameDetection,ProcTime,kPoint

from cv_bridge import CvBridge
import numpy as np

import pickle
import argparse

from statistics import mean,stdev
import matplotlib.pyplot as plt
import matplotlib.style as sty
cvb=CvBridge() 

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

