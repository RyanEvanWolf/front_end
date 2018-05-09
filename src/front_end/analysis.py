from front_end.features import descriptorLookUpTable,detectorLookUpTable
import numpy as np
from statistics import mean,stdev
import pickle

from cv_bridge import CvBridge

class featureDatabase:
    def __init__(self,pickleDir):
        self.table=detectorLookUpTable()
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
        Settings["Maximum"]=[]
        Settings["0.9Maximum"]=[]
        Settings["0.8Maximum"]=[]
        Settings["0.7Maximum"]=[]
        Settings["0.6Maximum"]=[]
        Settings["+Deviation"]=[]
        Settings["Mean"]=[]
        Settings["-Deviation"]=[]
        Settings["Minimum"]=[]
        Results={}
        Results["Maximum"]=[]
        Results["0.9Maximum"]=[]
        Results["0.8Maximum"]=[]
        Results["0.7Maximum"]=[]
        Results["0.6Maximum"]=[]
        Results["+Deviation"]=[]
        Results["Mean"]=[]
        Results["-Deviation"]=[]
        Results["Minimum"]=[]
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
    


# def getLandmarkStats(landmarkMsg):
#     ProcessingT
#     pass

# def getStereoStats(inputFrame,matcheframe):
#     print("")


def RMSerror(vector):
    RMS=0
    if(len(vector)>0):
        for v in vector:
            RMS+=np.power(v,2)
        RMS=np.sqrt(RMS/len(vector))
    return RMS

