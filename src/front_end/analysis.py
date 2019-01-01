# from front_end.features import descriptorLookUpTable,detectorLookUpTable
# from dataset.simulation import *
# from bumblebee.motion import *
# from bumblebee.baseTypes import *
# from dataset.utils import *
import numpy as np
from statistics import mean,stdev
import pickle
import cv2
from cv_bridge import CvBridge
import os
# import bumblebee.utils as butil
# import matplotlib.pyplot as plt 
# import matplotlib.patches as mpatches
# import matplotlib.style as sty
import rosbag

#sty.use("seaborn")
# class outlierData:
#     def __init__(self,d):
#         self.topDir=d
#     def getOperatingCurve(self):
#         l=os.listdir(self.topDir+"/outlier")
#         l.sort(key=float)
#         return l
#     def getNoiseLevels(self):
#         l=os.listdir(self.topDir+"/outlier/"+self.getOperatingCurve()[0])
#         l.sort(key=float)
#         return l
    
#     def getResultsAsList(self,opCurve,noise):
#         Hlist=sorted(os.listdir(self.topDir+"/outlier/"+opCurve+"/"+noise))
#         X=[]
#         RMS=[]
#         time=[]
#         inliers=[]
#         outliers=[]
#         for h in Hlist:
#             fname=self.topDir+"/outlier/"+opCurve+"/"+noise+"/"+h
#             with open(fname,"rb") as data_file:
#                 data_loaded = msgpack.unpack(data_file)
#                 currentFile = slidingWindow()
#                 currentFile.deserializeWindow(data_loaded[0])
#                 X.append(copy.deepcopy(currentFile.X[0:6,0]))
#                 RMS.append(data_loaded[1])
#                 time.append(data_loaded[2])
#                 inliers.append(len(currentFile.inliers))
#                 outliers.append(currentFile.nLandmarks-inliers[-1])
                
#         return X,RMS,time,inliers,outliers

def getTopic(rosbagFile,topic,Normalize=True):
    bag = rosbag.Bag(rosbagFile)
    outData=[]
    for topic, msg, t in bag.read_messages(topics=[topic]):
        outData.append(msg.data)
    bag.close()    

    return outData


# class simulationAnalyser:
#     def __init__(self,topDir,extractType="PCL",motionType="straight"):
#         self.topDir=topDir
#         self.motion=motionType
#         self.extractor=extractType
#     def getIdealMotionX(self,speedCat):
#         '''
#         Get the ideal motion X values
#         '''
#         motionX=[]
#         for h in self.getIdealHList():
#             currentFileDir=os.path.join(self.topDir,speedCat,self.motion,"Data",h)
#             with open(currentFileDir) as data_file:
#                 data_loaded = msgpack.unpack(data_file)
#                 currentFile = simulatedDataFrame(getCam=False)
#                 currentFile.deserializeFrame(data_loaded)
#                 motionX.append(currentFile.idealMotion)
#         return motionX
#     def getOutlierMotionX(self,speedCat,opCurve,outCat):
#         '''
#         get the estimated outlier motion 
#         for speed category, and outlier category percentage
#         PCL or BA estimator
#         '''
#         motionX=[]
#         for h in self.getIdealHList():
#             currentFileDir=os.path.join(self.topDir,speedCat,self.motion,self.extractor,"outlier",opCurve,outCat,h)
#             with open(currentFileDir) as data_file:
#                 abcd=msgpack.unpack(data_file)
#                 currentFile=slidingWindow()
#                 currentFile.deserializeWindow(abcd[0])
#                 motionX.append(currentFile.getPoseX(1))
#         return motionX
#     def generateMotionScatter(self,opCurve,noiseType,noiseLevel):
#         '''
#         plot the motion scatter with error
#         estimator=PCL or BA
#         opCurve= operational curve selected for plotting
#         noisecateg=outlier/outlier_curve || gaussian/gaussianNoise
#         '''
#         ########
#         ##draw the ideal motions
#         if(noiseType=="gaussian"):
#             folders=getGaussianFolderLevels()
#             colours=[np.random.rand(3) for i in folders]
#         else:
#             folders=getOutlierFolderLevels()
#             colours=[np.random.rand(3) for i in folders]
#         fig = plt.figure()
#         ax1 = fig.add_subplot(111)
#         colours=[np.random.rand(3) for i in getOutlierFolderLevels()]
        
#         for spd in ["Slow","Medium","Fast"]:
#             for op in getOperatingCurveFolderLevels():
                    
#                 for j in range(len(folders)):

#                     idealXVals=self.getIdealMotionX(spd)
#                     oval=np.random.rand(3)

#                     Rx=[getL2Formatted(getxPoseFormatted(copy.deepcopy(x),True))[0] for x in idealXVals]
#                     Rtheta=[getL2Formatted(getxPoseFormatted(copy.deepcopy(x),True))[1] for x in idealXVals]


                    
#                     if(noiseType=="gaussian"):
#                         outlierXVals=self.getGaussianMotionX(spd,op,folders[j])
#                     else:
#                         outlierXVals=self.getOutlierMotionX(spd,op,folders[j])
#                     grossFail=0
#                     for i in range(len(idealXVals)):
#                         t,r=getL2Formatted(getxPoseFormatted(copy.deepcopy(idealXVals[i])-copy.deepcopy(outlierXVals[i]),True))
#                         if(t>15 or r>15):
#                             grossFail+=1
#                         else:

#                             e=mpatches.Ellipse(xy=(Rtheta[i],Rx[i]),
#                             width=r, height=t,
#                             angle=0)
#                             ax1.add_artist(e)
#                             e.set_clip_box(ax1.bbox)
#                             e.set_alpha(0.6)
#                             e.set_facecolor(colours[j])
#                     print(spd,op,folders[j],grossFail)

#         for spd in ["Slow","Medium","Fast"]:
#             idealXVals=self.getIdealMotionX(spd)
#             Rx=[getL2Formatted(getxPoseFormatted(x,True))[0] for x in idealXVals]
#             Rtheta=[getL2Formatted(getxPoseFormatted(x,True))[1] for x in idealXVals]
#             ax1.scatter(Rtheta,Rx,s=8,c=(0,0,0,1),zorder=10)
#         # idealXVals=self.getIdealMotionX("Medium")

#         # Rx=[getL2Formatted(getxPoseFormatted(x,True))[0] for x in idealXVals]
#         # Rtheta=[getL2Formatted(getxPoseFormatted(x,True))[1] for x in idealXVals]

#         # ax1.scatter(Rtheta,Rx,s=1,c=(0,0,0,1))
#         # idealXVals=self.getIdealMotionX("Fast")

#         # Rx=[getL2Formatted(getxPoseFormatted(x,True))[0] for x in idealXVals]
#         # Rtheta=[getL2Formatted(getxPoseFormatted(x,True))[1] for x in idealXVals]

#         # ax1.scatter(Rtheta,Rx,s=1,c=(0,0,0,1))       

#         ##########
#         ##draw the outlier values




# # NUM = 250

# # ells = [Ellipse(xy=np.random.rand(2) * 10,
# #                 width=np.random.rand(), height=np.random.rand(),
# #                 angle=np.random.rand() * 360)
# #         for i in range(NUM)]

# # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
# # for e in ells:
# #     ax.add_artist(e)
# #     e.set_clip_box(ax.bbox)
# #     e.set_alpha(np.random.rand())
# #     e.set_facecolor(np.random.rand(3))

# # ax.set_xlim(0, 10)
# # ax.set_ylim(0, 10)

# # plt.show()



#             # for i in range(0,len(idealXVals)):
#             #     e=idealXVals[i]-outlierXVals[i]
#             #     e1=idealXVals[i]-gaussianXVals[i]
#             #     form=getxPoseFormatted(e,True)
#             #     form1=getxPoseFormatted(e1,True)
#             #     eT.append(np.sqrt((form["C"]**2).sum()))
#             #     eR.append(np.sqrt(form["Roll"]**2 +form["Yaw"]**2 +form["Pitch"]**2))            


#     def getGaussianMotionX(self,speedCat,opCurve,gaussCat):
#         motionX=[]
#         for h in self.getIdealHList():
#             currentFileDir=os.path.join(self.topDir,speedCat,self.motion,self.extractor,"gaussian",opCurve,gaussCat,h)
#             with open(currentFileDir) as data_file:
#                 abcd=msgpack.unpack(data_file)
#                 currentFile=slidingWindow()
#                 currentFile.deserializeWindow(abcd[0])
#                 motionX.append(currentFile.getPoseX(1))
#         return motionX
#     def getIdealHList(self):
#         return sorted(os.listdir(self.topDir+"/Slow/"+self.motion+"/Data"))
#     def getMotionError(self,speedCat,noiseCat,noiseLevel):
#         pass
#     def generateGrossFailure(self):

#         a=getGaussianFolderLevels()
#         for h in self.getIdealHList()[0:2]:
#             currentFileDir=os.path.join(self.topDir,"Slow","straight","PCL","gaussian","10",a[0],h)
#             print(currentFileDir)
#             with open(currentFileDir) as data_file:
#                 abcd=msgpack.unpack(data_file)   

        
#         return a
#         # params=7
#         # mot=3
#         # curves=4

#         # t=5
#         # fig, ax= plt.subplots(4, 3)
#         # for spd in range(mot):
#         #     if(spd==0):
#         #         c=plt.cm.get_cmap('Reds')
#         #     elif(spd==1):
#         #         c=plt.cm.get_cmap('Greens')
#         #     else:
#         #         c=plt.cm.get_cmap('Oranges')
#         # for a in ax:
#         #     for b in a:
#         #         c = b.pcolor(255*np.ones((40,7)),cmap=plt.cm.get_cmap('Greens'),edgecolors='k',linewidths=1)
#         #         c = b.pcolor(np.random.rand(np.random.randint(0,40),np.random.randint(0,7)),cmap=plt.cm.get_cmap('Reds'))
#         #         b.axis('off')
#         # print(len(ax))
#         # print(len(ax[0]))



#             # for opc in range(curves):
#             #     data=np.random.rand(t,7)
#             #     print(range(opc*7,opc*7+7))
#             #     c = ax0.pcolor(range(opc*7,opc*7+7),range(spd*t,spd*t+t),data,edgecolors='k',linewidths=1)
#         # Z=np.random.rand(40,3*4*3*4)

#         # fig, (ax0) = plt.subplots(1, 1)

#         # for i in range(0,10):
#         #     for j in range(0,40):
#         #         c = ax0.pcolor(range(i*3,i*3+3),range(j*4,j*4+4),np.random.rand(4,3),edgecolors='k',linewidths=1,cmap=plt.cm.get_cmap('Reds'))
#         #     ax0.axhline(3*i)
#         # c = ax0.pcolor(range(3*4*3*4),range(40,80),Z,edgecolors='k',linewidths=1,cmap=plt.cm.get_cmap('Blues'))
#         # c = ax0.pcolor(range(3*4*3*4),range(80,120),Z,edgecolors='k',linewidths=1,cmap=plt.cm.get_cmap('Greens'))
#         # c = ax0.pcolor(range(40),range(90,105),np.ones((15,40)),edgecolors='k',linewidths=1,cmap=plt.cm.get_cmap('Oranges'))
#         # z2=np.hstack((np.zeros((40,3*4*3*4)),Z))
#         # b=ax0.pcolor(z2,edgecolors='k',linewidths=1,cmap=plt.cm.get_cmap('Blues'))
#         # ax0.set_title('default: no edges')

#         fig.tight_layout()
#         plt.show()
# # class simulationAnalyser:
# #     def __init__(self,rootDir,spd="Slow",extractionMethod="BA",motionType="straight"):
# #         self.rootDir=rootDir
# #         self.extractMethod=extractionMethod
# #         self.type=motionType
# #         self.speed=spd
# #     def getDataFolder(self):
# #         return self.rootDir+"/"+self.speed+"/"+self.type+"/Data"
# #     def getExtractedFolder(self):
# #         return self.rootDir+"/"+self.speed +"/"+self.type+"/"+self.extractMethod
# #     def getMotionNames(self):
# #         return os.listdir(self.getDataFolder())
# #     def getGroupMotionStats(self):
# #         '''
# #         Categorize by the operational curve such that
# #         each result shows the operation curve + the outlier or noise level
# #         associated with each experiment.
# #         This can then be used to create a violin  plot.
# #         '''
# #         groupResults={}
# #         groupResults["noise"]={}
# #         groupResults["outlier"]={}
        
# #         simSettings=butil.getPickledObject(self.rootDir+"/landmark.p")
# #         OperatingCurves=os.listdir(self.getExtractedFolder()+"/ideal")

# #         params=("X","Y","Z","Roll","Pitch","Yaw")
# #         for o in params:
# #             groupResults["noise"][o]={}
# #             groupResults["outlier"][o]={}

# #             for c in OperatingCurves:
# #                 groupResults["noise"][o][c.lstrip("0")]={}
# #                 for levels in simSettings["GaussianNoise"]:
# #                     groupResults["noise"][o][c.lstrip("0")][str(levels)]=[]
# #                 groupResults["noise"][o][c.lstrip("0")]["ideal"]=[]
# #                 groupResults["outlier"][o][c.lstrip("0")]={}
# #                 for levels in simSettings["OutlierLevels"]:
# #                     groupResults["outlier"][o][c.lstrip("0")][str(int(levels*100))]=[]
# #                     print(o,c.lstrip("0"),str(int(levels*100)))
# #                 groupResults["outlier"][o][c.lstrip("0")]["ideal"]=[]
# #         motionFiles=self.getMotionNames()
# #         for motionIndex in motionFiles:
# #             print(motionIndex)
# #             singleError=self.getSingleMotionStats(motionIndex)
# #             ###add the ideal curves
# #             for j in singleError["ideal"].keys():
# #                 for p in params:
# #                     groupResults["noise"][p][j]["ideal"].append(singleError["ideal"][j][p])
# #                     groupResults["outlier"][p][j]["ideal"].append(singleError["ideal"][j][p])
            
# #             for j in singleError["outlier"].keys():
# #                 for p in params:
# #                     for l in groupResults["outlier"][p][j].keys():
# #                         if(l!="ideal"):
# #                             groupResults["outlier"][p][j][l].append(singleError["outlier"][j][l][p])
# #             for j in singleError["noise"].keys():
# #                 for p in params:
# #                     for l in groupResults["noise"][p][j].keys():
# #                         if(l!="ideal"):
# #                             groupResults["noise"][p][j][l].append(singleError["noise"][j][l][p])
# #         return groupResults
# #     def getSingleMotionStats(self,fileName):
# #         originalFile=butil.getPickledObject(self.getDataFolder()+"/"+fileName)
# #         results={}
# #         results["ideal"]={}
# #         results["noise"]={}
# #         results["outlier"]={}
# #         ###########
# #         ###Get the Ideal Data
# #         OperatingCurves=os.listdir(self.getExtractedFolder()+"/ideal")
# #         for currentOperatingCurve in OperatingCurves:
# #             #####
# #             ##Get the Ideal Data
# #             results["ideal"][currentOperatingCurve.lstrip("0")]={}
# #             idealFile=butil.getPickledObject(self.getExtractedFolder()+"/ideal/"+currentOperatingCurve+"/"+fileName)
# #             ###gen e1
# #             results["ideal"][currentOperatingCurve.lstrip("0")]=compareAbsoluteMotion(originalFile.motionEdge,
# #                                                                     decomposeTransform(np.linalg.inv(idealFile)))
# #         ########
# #         ##get the Noisy Data
# #         ################
# #         OperatingCurves=os.listdir(self.getExtractedFolder()+"/noise")
# #         for currentOperatingCurve in OperatingCurves:
# #             #####
# #             ##Get the Noisy Data Curve 
# #             noiseLevels=os.listdir(self.getExtractedFolder()+"/noise/"+currentOperatingCurve)
# #             results["noise"][currentOperatingCurve.lstrip("0")]={}
# #             for noiseLevel in noiseLevels:
# #                 ####
# #                 ##extract data per noise level
# #                 results["noise"][currentOperatingCurve.lstrip("0")][noiseLevel.replace("_",".")]={}
# #                 extractedDataFile=butil.getPickledObject(self.getExtractedFolder()+"/noise/"+currentOperatingCurve+"/"+noiseLevel+"/"+fileName)
# #                 ###gen e1
# #                 results["noise"][currentOperatingCurve.lstrip("0")][noiseLevel.replace("_",".")]=compareAbsoluteMotion(originalFile.motionEdge,
# #                                                                         decomposeTransform(np.linalg.inv(extractedDataFile)))
# #         #########
# #         ###get the Outlier Data
# #         #################
# #         OperatingCurves=os.listdir(self.getExtractedFolder()+"/outlier")
# #         for currentOperatingCurve in OperatingCurves:
# #             #####
# #             ##Get the Outlier Data Curve 
# #             outlierLevels=os.listdir(self.getExtractedFolder()+"/outlier/"+currentOperatingCurve)
# #             results["outlier"][currentOperatingCurve.lstrip("0")]={}
# #             for outlierLevel in outlierLevels:
# #                 results["outlier"][currentOperatingCurve.lstrip("0")][outlierLevel]={}
# #                 extractedDataFile=butil.getPickledObject(self.getExtractedFolder()+"/outlier/"+currentOperatingCurve+"/"+outlierLevel+"/"+fileName)
# #                 ###gen e1
# #                 results["outlier"][currentOperatingCurve.lstrip("0")][outlierLevel]=compareAbsoluteMotion(originalFile.motionEdge,
# #                                                                         decomposeTransform(np.linalg.inv(extractedDataFile)))
# #         return results

# def getOperatingCurves(folderDir,defaultDetectorTable=""):
#     if(defaultDetectorTable==""):
#         table=getDetectorTable()
#     else:
#         table=getDetectorTable(defaultDetectorTable)
#     detectorType=folderDir.split("/")[-1]
#     Settings={}
#     Times={}
#     Results={}
#     for i in OperatingCurveIDs():
#         Settings[i]=[]
#         Results[i]=[]
#         Times[i]=[]
#     print(folderDir)
#     for fileName in os.listdir(folderDir):
#         ##open a single image detection results pickle object
#         f=open(folderDir+"/"+fileName,"r")
#         data=pickle.load(f)
#         f.close()
#         leftFeaturesNList=[]
#         processingTime=[]
#         ####
#         ####get a list of left features detcted and the times for each for the image
#         for singleDetectionResult in data.outputFrames:
#             leftFeaturesNList.append(singleDetectionResult.nLeft)
#             processingTime.append(singleDetectionResult.processingTime[0].seconds*1000)
#             print(singleDetectionResult.nLeft,singleDetectionResult.processingTime[0].seconds*1000)
#         ###
#         ###determine the statistics of the features detected
#         ### and add them to a list with a label
#         MaxInFrame=np.amax(leftFeaturesNList)
#         MinInFrame=np.amin(leftFeaturesNList)
#         MeanInFrame=mean(leftFeaturesNList)
#         dev=stdev(leftFeaturesNList)
#         dev_mean=MeanInFrame+dev
#         IdealPerformanceTotals=[("Maximum",MaxInFrame),
#                         ("0.9Maximum",0.9*MaxInFrame),
#                         ("0.8Maximum",0.8*MaxInFrame),
#                         ("+Deviation",MeanInFrame+dev),
#                         ("Mean",MeanInFrame),
#                         ("-Deviation",np.clip(MeanInFrame-dev,0,MaxInFrame)),
#                         ("Minimum",MinInFrame)]
#         #####
#         ###for each statistic, find the detectorID with the closest index
#         for i in IdealPerformanceTotals:
#             closestIndex=np.abs(np.array(leftFeaturesNList)-i[1]).argmin()
#             # print(i)
#             # print(leftFeaturesNList[closestIndex])
#             Settings[i[0]].append(data.outputFrames[closestIndex].detID)
#             Results[i[0]].append(leftFeaturesNList[closestIndex])
#             #print(fileName,data.outputFrames[closestIndex].processingTime[0].seconds*1000)
#             print(processingTime[closestIndex])
#             Times[i[0]].append(processingTime[closestIndex])
#             #Times[i[0]].append(data.outputFrames[closestIndex].processingTime[0].seconds*1000)##in milliSeconds
#     return Settings,Results,Times


# class featureDatabase:
#     def __init__(self,pickleDir):
#         self.table=getDetectorTable
#         inputPickle=open(pickleDir,"rb")
#         self.featurePickle=pickle.load(inputPickle)
#         inputPickle.close()
#     def getOperatingCurves(self,detectorName):
#         nImages=len(self.featurePickle)
#         table=detectorLookUpTable()
#         allSettings=table.keys()
#         # ###get setting indexes
#         # idList=[]
#         # for i in range(0,len(allSettings)):
#         #     if(table[allSettings[i]]["Name"]==detectorName):
#         #         idList.append(i)
#         # for i in idList:
#         #     print(i,table[allSettings[i]]["Name"])

#         Settings={}
#         Results={}
#         for i in OperatingCurveIDs():
#             Settings[i]=[]
#             Results[i]=[]
#         for imageIndex in self.featurePickle:
#             ###make a list of nFeatures
#             leftNFeatures=[]
#             settings=[]
#             for settingsIndex in imageIndex.outputFrames:
#                 if(table[settingsIndex.detID]["Name"]==detectorName):
#                     leftNFeatures.append(settingsIndex.nLeft)
#                     settings.append(settingsIndex.detID)
#             MaxInFrame=np.amax(leftNFeatures)
#             MinInFrame=np.amin(leftNFeatures)
#             MeanInFrame=mean(leftNFeatures)
#             dev=stdev(leftNFeatures)
#             dev_mean=MeanInFrame+dev
#             IdealPerformanceTotals=[("Maximum",MaxInFrame),
#                             ("0.9Maximum",0.9*MaxInFrame),
#                             ("0.8Maximum",0.8*MaxInFrame),
#                             ("0.7Maximum",0.7*MaxInFrame),
#                             ("0.6Maximum",0.6*MaxInFrame),
#                             ("+Deviation",MeanInFrame+dev),
#                             ("Mean",MeanInFrame),
#                             ("-Deviation",np.clip(MeanInFrame-dev,0,MaxInFrame)),
#                             ("Minimum",MinInFrame)]
#             for i in IdealPerformanceTotals:
#                 closestIndex=np.abs(np.array(leftNFeatures)-i[1]).argmin()
#                 print(i)
#                 print(leftNFeatures[closestIndex])
#                 Settings[i[0]].append(settings[closestIndex])
#                 Results[i[0]].append(leftNFeatures[closestIndex])
#         return Settings,Results
#     def getAllProcessingTime(self,detectorName):
#         nImages=len(self.featurePickle)
#         table=detectorLookUpTable()
#         allSettings=table.keys()
#         ###get setting indexes
#         idList=[]
#         for i in range(0,len(allSettings)):
#             if(table[allSettings[i]]["Name"]==detectorName):
#                 idList.append(i)
#         print("totalSettings",len(idList))
#         print("totalWithSurf",len(self.featurePickle),len(self.featurePickle[0].outputFrames))
#         lprocTime=[]
#         frameN=[]
#         for i in range(0,len(self.featurePickle)):
#             for feature in idList:
#                 frameN.append(i)
#                 lprocTime.append(self.featurePickle[i].outputFrames[feature].processingTime[0].seconds*1000)
#         return frameN,lprocTime
#     def getProcessingTime(self,Settings,Results):
#         pass


# def getStereoFrameStatistics(inputFrame,landmarkFrame):
#                         #stereoFeatures,stereoLandmarks
#     #####calculate stereo Matching Stats
#     ###epiPolar Error
#     Results={}
#     Results["ProcessingTime"]={}
#     epipolar_error=[]
#     for i in range(0,len(landmarkFrame.leftFeatures)):
#         epipolar_error.append(landmarkFrame.leftFeatures[i].y-
#                                 landmarkFrame.rightFeatures[i].y)
    
#     Results["Epi"]=RMSerror(epipolar_error)
#     ###inlier Ratio
#     print("Frames Info:",len(landmarkFrame.leftFeatures),len(inputFrame.leftFeatures))
#     try:
#         Results["InlierRatio"]=float(len(landmarkFrame.leftFeatures))/float(len(inputFrame.leftFeatures))
#     except:
#         Results["InlierRatio"]=0
#     Results["nMatches"]=len(landmarkFrame.leftFeatures)
#     ###processing Time
#     for i in inputFrame.proc:
#         Results["ProcessingTime"][i.label]=i.seconds
#     for i in landmarkFrame.proc:
#         Results["ProcessingTime"][i.label]=i.seconds
#     return Results

# def getWindowStateStatistics(windowState,leftInfo,rightInfo,Q):
#     cvb=CvBridge()
#     Results={}
#     ###get nInliers
#     print("length",len(windowState.tracks))
#     for i in windowState.tracks:
#         img=cvb.imgmsg_to_cv2(i.motionInliers)
#         Results["nTracks"]=np.count_nonzero(img)
#     ###get inlierRatio
#     print(len(windowState.tracks))
#     print(len(windowState.tracks[0].tracks))
#     Results["InlierRatio"]=float(Results["nTracks"])/float(len(windowState.tracks[0].tracks))
#     ###get H
#     ###
#     ###current Points
    
#     ###previous Points
#     return Results

# def RMSerror(vector):
#     RMS=0
#     if(len(vector)>0):
#         for v in vector:
#             RMS+=np.power(v,2)
#         RMS=np.sqrt(RMS/len(vector))
#     return RMS

