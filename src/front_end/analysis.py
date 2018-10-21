from front_end.features import descriptorLookUpTable,detectorLookUpTable
from front_end.motion import *
from dataset.utils import *
import numpy as np
from statistics import mean,stdev
import pickle
import cv2
from cv_bridge import CvBridge
import os
import bumblebee.utils as butil


class simulationAnalyser:
    def __init__(self,rootDir,spd="Slow",extractionMethod="BA",motionType="straight"):
        self.rootDir=rootDir
        self.extractMethod=extractionMethod
        self.type=motionType
        self.speed=spd
    def getDataFolder(self):
        return self.rootDir+"/"+self.speed+"/"+self.type+"/Data"
    def getExtractedFolder(self):
        return self.rootDir+"/"+self.speed +"/"+self.type+"/"+self.extractMethod
    def getMotionNames(self):
        return os.listdir(self.getDataFolder())
    def getGroupMotionStats(self):
        '''
        Categorize by the operational curve such that
        each result shows the operation curve + the outlier or noise level
        associated with each experiment.
        This can then be used to create a violin  plot.
        '''
        groupResults={}
        groupResults["noise"]={}
        groupResults["outlier"]={}
        
        simSettings=butil.getPickledObject(self.rootDir+"/landmark.p")
        OperatingCurves=os.listdir(self.getExtractedFolder()+"/ideal")

        params=("X","Y","Z","Roll","Pitch","Yaw")
        for o in params:
            groupResults["noise"][o]={}
            groupResults["outlier"][o]={}

            for c in OperatingCurves:
                groupResults["noise"][o][c.lstrip("0")]={}
                for levels in simSettings["GaussianNoise"]:
                    groupResults["noise"][o][c.lstrip("0")][str(levels)]=[]
                groupResults["noise"][o][c.lstrip("0")]["ideal"]=[]
                groupResults["outlier"][o][c.lstrip("0")]={}
                for levels in simSettings["OutlierLevels"]:
                    groupResults["outlier"][o][c.lstrip("0")][str(int(levels*100))]=[]
                    print(o,c.lstrip("0"),str(int(levels*100)))
                groupResults["outlier"][o][c.lstrip("0")]["ideal"]=[]
        motionFiles=self.getMotionNames()
        for motionIndex in motionFiles:
            print(motionIndex)
            singleError=self.getSingleMotionStats(motionIndex)
            ###add the ideal curves
            for j in singleError["ideal"].keys():
                for p in params:
                    groupResults["noise"][p][j]["ideal"].append(singleError["ideal"][j][p])
                    groupResults["outlier"][p][j]["ideal"].append(singleError["ideal"][j][p])
            
            for j in singleError["outlier"].keys():
                for p in params:
                    for l in groupResults["outlier"][p][j].keys():
                        if(l!="ideal"):
                            groupResults["outlier"][p][j][l].append(singleError["outlier"][j][l][p])
            for j in singleError["noise"].keys():
                for p in params:
                    for l in groupResults["noise"][p][j].keys():
                        if(l!="ideal"):
                            groupResults["noise"][p][j][l].append(singleError["noise"][j][l][p])
        return groupResults
    def getSingleMotionStats(self,fileName):
        originalFile=butil.getPickledObject(self.getDataFolder()+"/"+fileName)
        results={}
        results["ideal"]={}
        results["noise"]={}
        results["outlier"]={}
        ###########
        ###Get the Ideal Data
        OperatingCurves=os.listdir(self.getExtractedFolder()+"/ideal")
        for currentOperatingCurve in OperatingCurves:
            #####
            ##Get the Ideal Data
            results["ideal"][currentOperatingCurve.lstrip("0")]={}
            idealFile=butil.getPickledObject(self.getExtractedFolder()+"/ideal/"+currentOperatingCurve+"/"+fileName)
            ###gen e1
            results["ideal"][currentOperatingCurve.lstrip("0")]=compareAbsoluteMotion(originalFile.motionEdge,
                                                                    decomposeTransform(np.linalg.inv(idealFile)))
        ########
        ##get the Noisy Data
        ################
        OperatingCurves=os.listdir(self.getExtractedFolder()+"/noise")
        for currentOperatingCurve in OperatingCurves:
            #####
            ##Get the Noisy Data Curve 
            noiseLevels=os.listdir(self.getExtractedFolder()+"/noise/"+currentOperatingCurve)
            results["noise"][currentOperatingCurve.lstrip("0")]={}
            for noiseLevel in noiseLevels:
                ####
                ##extract data per noise level
                results["noise"][currentOperatingCurve.lstrip("0")][noiseLevel.replace("_",".")]={}
                extractedDataFile=butil.getPickledObject(self.getExtractedFolder()+"/noise/"+currentOperatingCurve+"/"+noiseLevel+"/"+fileName)
                ###gen e1
                results["noise"][currentOperatingCurve.lstrip("0")][noiseLevel.replace("_",".")]=compareAbsoluteMotion(originalFile.motionEdge,
                                                                        decomposeTransform(np.linalg.inv(extractedDataFile)))
        #########
        ###get the Outlier Data
        #################
        OperatingCurves=os.listdir(self.getExtractedFolder()+"/outlier")
        for currentOperatingCurve in OperatingCurves:
            #####
            ##Get the Outlier Data Curve 
            outlierLevels=os.listdir(self.getExtractedFolder()+"/outlier/"+currentOperatingCurve)
            results["outlier"][currentOperatingCurve.lstrip("0")]={}
            for outlierLevel in outlierLevels:
                results["outlier"][currentOperatingCurve.lstrip("0")][outlierLevel]={}
                extractedDataFile=butil.getPickledObject(self.getExtractedFolder()+"/outlier/"+currentOperatingCurve+"/"+outlierLevel+"/"+fileName)
                ###gen e1
                results["outlier"][currentOperatingCurve.lstrip("0")][outlierLevel]=compareAbsoluteMotion(originalFile.motionEdge,
                                                                        decomposeTransform(np.linalg.inv(extractedDataFile)))
        return results

def getOperatingCurves(folderDir,defaultDetectorTable=""):
    if(defaultDetectorTable==""):
        table=getDetectorTable()
    else:
        table=getDetectorTable(defaultDetectorTable)
    detectorType=folderDir.split("/")[-1]
    Settings={}
    Times={}
    Results={}
    for i in OperatingCurveIDs():
        Settings[i]=[]
        Results[i]=[]
        Times[i]=[]
    print(folderDir)
    for fileName in os.listdir(folderDir):
        ##open a single image detection results pickle object
        f=open(folderDir+"/"+fileName,"r")
        data=pickle.load(f)
        f.close()
        leftFeaturesNList=[]
        processingTime=[]
        ####
        ####get a list of left features detcted and the times for each for the image
        for singleDetectionResult in data.outputFrames:
            leftFeaturesNList.append(singleDetectionResult.nLeft)
            processingTime.append(singleDetectionResult.processingTime[0].seconds*1000)
            print(singleDetectionResult.nLeft,singleDetectionResult.processingTime[0].seconds*1000)
        ###
        ###determine the statistics of the features detected
        ### and add them to a list with a label
        MaxInFrame=np.amax(leftFeaturesNList)
        MinInFrame=np.amin(leftFeaturesNList)
        MeanInFrame=mean(leftFeaturesNList)
        dev=stdev(leftFeaturesNList)
        dev_mean=MeanInFrame+dev
        IdealPerformanceTotals=[("Maximum",MaxInFrame),
                        ("0.9Maximum",0.9*MaxInFrame),
                        ("0.8Maximum",0.8*MaxInFrame),
                        ("+Deviation",MeanInFrame+dev),
                        ("Mean",MeanInFrame),
                        ("-Deviation",np.clip(MeanInFrame-dev,0,MaxInFrame)),
                        ("Minimum",MinInFrame)]
        #####
        ###for each statistic, find the detectorID with the closest index
        for i in IdealPerformanceTotals:
            closestIndex=np.abs(np.array(leftFeaturesNList)-i[1]).argmin()
            # print(i)
            # print(leftFeaturesNList[closestIndex])
            Settings[i[0]].append(data.outputFrames[closestIndex].detID)
            Results[i[0]].append(leftFeaturesNList[closestIndex])
            #print(fileName,data.outputFrames[closestIndex].processingTime[0].seconds*1000)
            print(processingTime[closestIndex])
            Times[i[0]].append(processingTime[closestIndex])
            #Times[i[0]].append(data.outputFrames[closestIndex].processingTime[0].seconds*1000)##in milliSeconds
    return Settings,Results,Times


class featureDatabase:
    def __init__(self,pickleDir):
        self.table=getDetectorTable
        inputPickle=open(pickleDir,"rb")
        self.featurePickle=pickle.load(inputPickle)
        inputPickle.close()
    def getOperatingCurves(self,detectorName):
        nImages=len(self.featurePickle)
        table=detectorLookUpTable()
        allSettings=table.keys()
        # ###get setting indexes
        # idList=[]
        # for i in range(0,len(allSettings)):
        #     if(table[allSettings[i]]["Name"]==detectorName):
        #         idList.append(i)
        # for i in idList:
        #     print(i,table[allSettings[i]]["Name"])

        Settings={}
        Results={}
        for i in OperatingCurveIDs():
            Settings[i]=[]
            Results[i]=[]
        for imageIndex in self.featurePickle:
            ###make a list of nFeatures
            leftNFeatures=[]
            settings=[]
            for settingsIndex in imageIndex.outputFrames:
                if(table[settingsIndex.detID]["Name"]==detectorName):
                    leftNFeatures.append(settingsIndex.nLeft)
                    settings.append(settingsIndex.detID)
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
                print(i)
                print(leftNFeatures[closestIndex])
                Settings[i[0]].append(settings[closestIndex])
                Results[i[0]].append(leftNFeatures[closestIndex])
        return Settings,Results
    def getAllProcessingTime(self,detectorName):
        nImages=len(self.featurePickle)
        table=detectorLookUpTable()
        allSettings=table.keys()
        ###get setting indexes
        idList=[]
        for i in range(0,len(allSettings)):
            if(table[allSettings[i]]["Name"]==detectorName):
                idList.append(i)
        print("totalSettings",len(idList))
        print("totalWithSurf",len(self.featurePickle),len(self.featurePickle[0].outputFrames))
        lprocTime=[]
        frameN=[]
        for i in range(0,len(self.featurePickle)):
            for feature in idList:
                frameN.append(i)
                lprocTime.append(self.featurePickle[i].outputFrames[feature].processingTime[0].seconds*1000)
        return frameN,lprocTime
    def getProcessingTime(self,Settings,Results):
        pass


def getStereoFrameStatistics(inputFrame,landmarkFrame):
                        #stereoFeatures,stereoLandmarks
    #####calculate stereo Matching Stats
    ###epiPolar Error
    Results={}
    Results["ProcessingTime"]={}
    epipolar_error=[]
    for i in range(0,len(landmarkFrame.leftFeatures)):
        epipolar_error.append(landmarkFrame.leftFeatures[i].y-
                                landmarkFrame.rightFeatures[i].y)
    
    Results["Epi"]=RMSerror(epipolar_error)
    ###inlier Ratio
    print("Frames Info:",len(landmarkFrame.leftFeatures),len(inputFrame.leftFeatures))
    try:
        Results["InlierRatio"]=float(len(landmarkFrame.leftFeatures))/float(len(inputFrame.leftFeatures))
    except:
        Results["InlierRatio"]=0
    Results["nMatches"]=len(landmarkFrame.leftFeatures)
    ###processing Time
    for i in inputFrame.proc:
        Results["ProcessingTime"][i.label]=i.seconds
    for i in landmarkFrame.proc:
        Results["ProcessingTime"][i.label]=i.seconds
    return Results

def getWindowStateStatistics(windowState,leftInfo,rightInfo,Q):
    cvb=CvBridge()
    Results={}
    ###get nInliers
    print("length",len(windowState.tracks))
    for i in windowState.tracks:
        img=cvb.imgmsg_to_cv2(i.motionInliers)
        Results["nTracks"]=np.count_nonzero(img)
    ###get inlierRatio
    print(len(windowState.tracks))
    print(len(windowState.tracks[0].tracks))
    Results["InlierRatio"]=float(Results["nTracks"])/float(len(windowState.tracks[0].tracks))
    ###get H
    ###
    ###current Points
    
    ###previous Points
    return Results

def RMSerror(vector):
    RMS=0
    if(len(vector)>0):
        for v in vector:
            RMS+=np.power(v,2)
        RMS=np.sqrt(RMS/len(vector))
    return RMS

