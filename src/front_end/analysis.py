from front_end.features import descriptorLookUpTable,detectorLookUpTable
import numpy as np
from statistics import mean,stdev
import pickle

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
        ###get setting indexes
        idList=[]
        for i in range(0,len(allSettings)):
            if(table[allSettings[i]]["Name"]==detectorName):
                idList.append(i)
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
            leftNFeatures=[]
            for feature in idList:
                leftNFeatures.append(imageIndex.outputFrames[feature].nLeft)
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
                Settings[i[0]].append(allSettings[closestIndex])
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
