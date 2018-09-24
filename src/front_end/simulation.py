import cv2
import math
import time
import random
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_from_euler, quaternion_matrix
from front_end.motion import *
import matplotlib.pyplot as plt
import copy

import os

import pickle

noiseLevels={}
noiseLevels["0_25"]=0.25
noiseLevels["0_5"]=0.5
noiseLevels["0_75"]=0.75
noiseLevels["1"]=1
noiseLevels["1_5"]=1.5
noiseLevels["2"]=2
noiseLevels["2_5"]=2.5
outlierLevels=[0.05,0.1,0.15,0.2,0.25]

class simDirectory:
    def __init__(self,rootDir):
        self.root=rootDir
    def getSettings(self):
        m=pickle.load(open(self.root+"/motion.p"))
        c=pickle.load(open(self.root+"/camera.p"))
        n=pickle.load(open(self.root+"/Nister.p"))
        return c,m,n
    def getIdealWorldDir(self):
        s=self.root+"/slow_ideal"
        m=self.root+"/medium_ideal"
        f=self.root+"/fast_ideal"
        return s,m,f
    def getNoisyWorldDir(self):
        s=self.root+"/slow_noisy"
        m=self.root+"/medium_noisy"
        f=self.root+"/fast_noisy"
        return s,m,f     
    def getOutlierDir(self):
        s=self.root+"/slow_outlier"
        m=self.root+"/medium_outlier"
        f=self.root+"/fast_outlier"
        return s,m,f
def MotionCategorySettings():
    Settings={}
    Settings["Fast"]={}
    Settings["Medium"]={}
    Settings["Slow"]={}
    Settings["Fast"]["TranslationMean"]=0.066
    Settings["Fast"]["RotationMean"]=0
    Settings["Fast"]["TranslationNoise"]=0.1*Settings["Fast"]["TranslationMean"] ##meters
    Settings["Fast"]["RotationNoise"]=8      ##degrees

    Settings["Medium"]["TranslationMean"]=0.044
    Settings["Medium"]["RotationMean"]=0
    Settings["Medium"]["TranslationNoise"]=0.1*Settings["Medium"]["TranslationMean"] ##meters
    Settings["Medium"]["RotationNoise"]=4        ##degrees

    Settings["Slow"]["TranslationMean"]=0.022
    Settings["Slow"]["RotationMean"]=0
    Settings["Slow"]["TranslationNoise"]=0.1*Settings["Slow"]["TranslationMean"] ##meters
    Settings["Slow"]["RotationNoise"]=1        ##degrees
    return Settings

def getSimulatedLandmarkSettings():
    Settings={}
    Settings["Xdepth"]=5.0
    Settings["Ydepth"]=5.0
    Settings["Zdepth"]=4.0
    Settings["HeightMinimum"]=0.5
    Settings["MinimumOutlier"]=3.0 #pixels
    Settings["OutlierLevels"]=[0.05,0.1,0.15,0.2,0.25]
    Settings["GaussianNoise"]=[0.25,0.5,0.75,1.0,1.5,2,2.5]
    Settings["operatingCurves"]=[0.3,0.5,0.7,0.9,1.0]
    return Settings

def getCameraSettingsFromServer():
    cvb=CvBridge()
    ##assumes a node has been declared
    cameraSettings={}
    cameraSettings["Q"]=cvb.imgmsg_to_cv2(rospy.wait_for_message("/bumblebee_configuration/Q",Image))
    cameraSettings["lInfo"]=rospy.wait_for_message("/bumblebee_configuration/ideal/leftRectified/CameraInfo",CameraInfo)
    cameraSettings["rInfo"]=rospy.wait_for_message("/bumblebee_configuration/ideal/rightRectified/CameraInfo",CameraInfo)
    cameraSettings["Pl"]=np.zeros((3,4),dtype=np.float64)
    cameraSettings["Pr"]=np.zeros((3,4),dtype=np.float64)
    for row in range(0,3):
            for col in range(0,4):
                cameraSettings["Pl"][row,col]=cameraSettings["lInfo"].P[row*4 +col]
                cameraSettings["Pr"][row,col]=cameraSettings["rInfo"].P[row*4 +col]

    cameraSettings["width"]=cameraSettings["lInfo"].width
    cameraSettings["height"]=cameraSettings["lInfo"].height
    cameraSettings["f"]=cameraSettings["Pl"][0,0]
    cameraSettings["pp"]=(cameraSettings["Pl"][0:2,2][0],
                        cameraSettings["Pl"][0:2,2][1])
    cameraSettings["k"]=cameraSettings["Pl"][0:3,0:3]
    print("Loaded")
    return cameraSettings

def noisyRotations(noise=5):
    frame={}
    frame["Roll"]=np.random.normal(0,noise,1)
    frame["Pitch"]=np.random.normal(0,noise,1)
    frame["Yaw"]=np.random.normal(0,noise,1)
    q=quaternion_from_euler(math.radians(frame["Roll"]),
                            math.radians(frame["Pitch"]),
                            math.radians(frame["Yaw"]),'szxy')
    frame["matrix"]=quaternion_matrix(q)[0:3,0:3]    
    return frame

def steeringRotation():
    print("a")

def dominantTranslation(zBase=0.2,noise=0.1):
    frame={}
    frame["X"]=np.random.normal(0,noise,1)
    frame["Y"]=np.random.normal(0,noise,1)
    frame["Z"]=abs(np.random.normal(zBase,noise,1))
    t=np.zeros((3,1),dtype=np.float64)
    t[0,0]=frame["X"]
    t[1,0]=frame["Y"]
    t[2,0]=frame["Z"]
    frame["vector"]=t
    return frame

def genDefaultNisterSettings(cameraConfig):
    settings={}
    settings["Pl"]=cameraConfig["Pl"]
    settings["Pr"]=cameraConfig["Pr"]
    settings["pp"]=cameraConfig["pp"]
    settings["k"]=cameraConfig["k"]
    settings["f"]=cameraConfig["f"]
    settings["threshold"]=3
    settings["probability"]=0.99
    return settings

def genDefaultStraightSimulationConfig(Pl,Pr,Q,width,height):
    Settings={}
    Settings["FastDeviation"]=0.3
    Settings["MediumDeviation"]=0.2
    Settings["SlowDeviation"]=0.1
    Settings["TranslationNoise"]=0.1
    Settings["RotationNoise"]=1
    Settings["noise"]=np.linspace(0,20,5,dtype=np.float64)
    Settings["outlierPercentage"]=np.linspace(10,40,5)
    Settings["TotalSimulations"]=1
    Settings["TotalPoints"]=300
    Settings["Pl"]=Pl
    Settings["Pr"]=Pr
    Settings["Q"]=Q
    Settings["width"]=width
    Settings["height"]=height
    Settings["blankImage"]=np.zeros((height,
                                     width,3),dtype=np.uint8)
    return Settings

# class nisterExtract:
#     def __init__(self,rootDir,extractConfig):
#         self.root=rootDir
#         self.output=rootDir+"/Nister"
#         self.extract=extractConfig
#     def extractMotion(self,inputFolder):
#         worldFilesSet=os.listdir(self.root+"/"+inputFolder)
#         print("Loaded From "+self.root+"/"+inputFolder)
#         for Hpickle in worldFilesSet:
#             f=open(self.root+"/"+inputFolder+"/"+Hpickle,"r")
#             data=pickle.load(f)
#             f.close()
#             print(getMotion(data["H"]))
#             HResults={}
#             for curve in data["Curves"]:
#                 curveID=str(len(curve))
#                 HResults[curveID]={}
#                 simulationPoints=[]
#                 for pointIndex in curve:
#                     simulationPoints.append(data["Points"][pointIndex])
#                     newPts=np.zeros((len(simulationPoints),2),dtype=np.float64)
#                     oldPts=np.zeros((len(simulationPoints),2),dtype=np.float64)
#                 for j in range(0,len(simulationPoints)):
#                     newPts[j,0]=simulationPoints[j]["Lb"][0]
#                     newPts[j,1]=simulationPoints[j]["Lb"][1]
#                     oldPts[j,0]=simulationPoints[j]["La"][0]
#                     oldPts[j,1]=simulationPoints[j]["La"][1]
#                 E,mask=cv2.findEssentialMat(newPts,oldPts,self.extract["f"],self.extract["pp"])
#                                             #,prob=self.extract["probability"],threshold=self.extract["threshold"])#,threshold=1)    #
#                 nInliers,R,T,matchMask=cv2.recoverPose(E,newPts,oldPts,self.extract["k"],mask)
#                 averageScale=np.zeros((3,3),dtype=np.float64)
#                 countedIn=0
#                 for index in range(0,len(simulationPoints)):
#                     i=simulationPoints[index]
#                     if(matchMask[index,0]==255):
#                         scale=(i["Xa"][0:3,0]-R.dot(i["Xb"][0:3,0])).reshape(3,1).dot(np.transpose(T.reshape(3,1))).dot(np.linalg.pinv(T.dot(np.transpose(T))))
#                         averageScale+=scale 
#                         countedIn+=1
#                 averageScale=averageScale/nInliers
#                 T=averageScale.dot(T)  
#                 original=createHomog(R,T)
#                 HResults[curveID]["H"]=np.linalg.inv(original)
#                 print(getMotion(HResults[curveID]["H"]))
#                 HResults[curveID]["Motion"]=getMotion(HResults[curveID]["H"]) 
#                 HResults[curveID]["inlierMask"]=matchMask
#                 HResults[curveID]["nInlier"]=nInliers
#                 HResults[curveID]["inlierRatio"]=nInliers/float(len(simulationPoints))
#                 HResults[curveID]["E"]=E
#                 HResults[curveID]["MotionError"]=compareMotion(HResults[curveID]["H"],data["H"])
#                 HResults[curveID]["CurveID"]=len(simulationPoints)
#                 HResults[curveID]["PointResults"]=[]
#                 #####get reprojection results
#                 for index in range(0,len(simulationPoints)):
#                     i=simulationPoints[index]
#                     if(matchMask[index,0]==255):
#                         HResults[curveID]["PointResults"].append(self.getLandmarkReprojection(i,HResults[curveID]["H"]) )
#             f=open(self.output+"/"+inputFolder+"/"+Hpickle,"w")
#             pickle.dump(HResults,f)
#             f.close()
#             print("----")        
#     def getMotionError(self,H,Hestimate):
#         orig=getMotion(H)
#         est=getMotion(Hestimate)
#         out["X"]

#         angleError=math.sqrt((orig["Roll"]-est["Roll"])**2 +
#                              (orig["Yaw"]-est["Yaw"])**2 +
#                               (orig["Pitch"]-orig["Pitch"])**2)
#         TranslationError=math.sqrt((orig["X"]-est["X"])**2+
#                                     (orig["Y"]-est["Y"])**2+
#                                     (orig["Z"]-est["Z"])**2)
#         out={}
#         out["TranslationError"]=TranslationError
#         out["angleError"]=angleError
#         return out
#     def getLandmarkReprojection(self,simPoint,H):
#         simResult={}
#         la_estimate=self.extract["Pl"].dot(simPoint["Xa"])
#         la_estimate=la_estimate/la_estimate[2,0]
#         ra_estimate=self.extract["Pr"].dot(simPoint["Xa"])
#         ra_estimate=ra_estimate/ra_estimate[2,0]
#         Xb_estimate=H.dot(simPoint["Xa"])

#         lb_estimate=self.extract["Pl"].dot(Xb_estimate)
#         lb_estimate=lb_estimate/lb_estimate[2,0]
#         rb_estimate=self.extract["Pr"].dot(Xb_estimate)
#         rb_estimate=rb_estimate/rb_estimate[2,0]
#         simResult["La_Pred"]=la_estimate
#         simResult["Ra_Pred"]=ra_estimate
#         simResult["Lb_Pred"]=lb_estimate
#         simResult["Rb_Pred"]=rb_estimate
#         simResult["Error"]=[lb_estimate-simPoint["Lb"],
#                             la_estimate-simPoint["La"],
#                             ra_estimate-simPoint["Ra"],
#                             rb_estimate-simPoint["Rb"]]
#         return simResult

def checkROI(inPoint,width,height):
        if((inPoint[0]>0)and(inPoint[0]<width)):
            if((inPoint[1]>0)and(inPoint[1]<height)):
                return True
            else:
                return False
        else:
            return False

def checkValidSimulatedPoint(inPoint,imageWidth,imageHeight,minHeight):
    if(checkROI(inPoint["La"],imageWidth,imageHeight)
        and checkROI(inPoint["Ra"],imageWidth,imageHeight)
        and checkROI(inPoint["Lb"],imageWidth,imageHeight)
        and checkROI(inPoint["Rb"],imageWidth,imageHeight)
        and (inPoint["Xa"][1,0]<minHeight)
        and (inPoint["Xb"][1,0]<minHeight)
        and (inPoint["Xa"][2,0]>0)
        and (inPoint["Xb"][2,0]>0)):
        return True
    else:
        return False

class simulatedLandmark:
    def __init__(self,Ha2b,CameraConfig,landmarkSettings=getSimulatedLandmarkSettings()):
        validPoint=False
        while(not validPoint):
            self.Data={}    
            x=np.random.normal(0,5,1)
            y=np.random.normal(0,5,1)
            z=np.random.normal(0,4,1)
            Point=np.ones((4,1),dtype=np.float64)
            self.Data["Xa"]=copy.deepcopy(Point)
            self.Data["Xa"][0,0]=x
            self.Data["Xa"][1,0]=y
            self.Data["Xa"][2,0]=z
            self.Data["La"]=CameraConfig["Pl"].dot(self.Data["Xa"])
            self.Data["La"]=self.Data["La"]/self.Data["La"][2,0]
            self.Data["Ra"]=CameraConfig["Pr"].dot(self.Data["Xa"])
            self.Data["Ra"]=self.Data["Ra"]/self.Data["Ra"][2,0]
            ##transform to second coordinate system
            self.Data["Xb"]=np.dot(Ha2b,self.Data["Xa"])
            self.Data["Xb"]=self.Data["Xb"]/self.Data["Xb"][3,0]
            self.Data["Lb"]=CameraConfig["Pl"].dot(self.Data["Xb"])
            self.Data["Lb"]=self.Data["Lb"]/self.Data["Lb"][2,0]
            self.Data["Rb"]=CameraConfig["Pr"].dot(self.Data["Xb"])
            self.Data["Rb"]=self.Data["Rb"]/self.Data["Rb"][2,0]   
            if(checkValidSimulatedPoint(self.Data,CameraConfig["width"],
                                        CameraConfig["height"],
                                        landmarkSettings["HeightMinimum"])):
                validPoint=True
                #####
                ###create gaussian noise simulations
                self.Noise=[]
                for i in landmarkSettings["GaussianNoise"]:
                    validNoise=False
                    while(not validNoise):
                        noisyData=copy.deepcopy(self.Data)
                        noisyData["La"][0,0]+=np.random.normal(0,i,1)
                        noisyData["La"][1,0]+=np.random.uniform()
                        noisyData["Ra"][0,0]+=np.random.normal(0,i,1)
                        noisyData["Ra"][1,0]+=np.random.uniform()
                        ##re triangulate
                        noisyData["Xa"]=cv2.triangulatePoints(CameraConfig["Pl"],CameraConfig["Pr"],
                                                                (noisyData["La"][0,0],noisyData["La"][1,0]),
                                                                (noisyData["Ra"][0,0],noisyData["Ra"][1,0]))
                        noisyData["Xa"]/=noisyData["Xa"][3,0]
                        noisyData["Lb"][0,0]+=np.random.normal(0,i,1)
                        noisyData["Lb"][1,0]+=np.random.uniform()
                        noisyData["Rb"][0,0]+=np.random.normal(0,i,1)
                        noisyData["Rb"][1,0]+=np.random.uniform()
                        #re triangulate
                        noisyData["Xb"]=cv2.triangulatePoints(CameraConfig["Pl"],CameraConfig["Pr"],
                                        (noisyData["Lb"][0,0],noisyData["Lb"][1,0]),
                                        (noisyData["Rb"][0,0],noisyData["Rb"][1,0]))
                        noisyData["Xb"]/=noisyData["Xb"][3,0]
                        if(checkValidSimulatedPoint(noisyData,CameraConfig["width"],
                            CameraConfig["height"],
                            landmarkSettings["HeightMinimum"])):
                                validNoise=True
                                self.Noise.append(noisyData)
                validOutlier=False
                self.Outlier=copy.deepcopy(self.Data)
                while(not validOutlier):
                    x=np.random.uniform(0.0,CameraConfig["width"])
                    y=np.random.uniform(0.0,CameraConfig["height"])
                    pts=np.ones((3,1),dtype=np.float64)
                    pts[0,0]=x
                    pts[1,0]=y 
                    diff=abs(self.Data["La"]-pts) 
                    if((diff[0,0]>landmarkSettings["MinimumOutlier"])
                        and(diff[1,0]>landmarkSettings["MinimumOutlier"])):
                        validOutlier=True
                        self.Outlier["La"]=pts
                        self.Outlier["Ra"]=np.random.uniform(0.0,CameraConfig["width"])
                validOutlier=False
                while(not validOutlier):
                    x=np.random.uniform(0.0,CameraConfig["width"])
                    y=np.random.uniform(0.0,CameraConfig["height"])
                    pts=np.ones((3,1),dtype=np.float64)
                    pts[0,0]=x
                    pts[1,0]=y 
                    diff=abs(self.Data["Lb"]-pts) 
                    if((diff[0,0]>landmarkSettings["MinimumOutlier"])
                        and(diff[1,0]>landmarkSettings["MinimumOutlier"])):
                        validOutlier=True
                        self.Outlier["Lb"]=pts
                        self.Outlier["Rb"]=np.random.uniform(0.0,CameraConfig["width"])   
      
class structDataset:
    def __init__(self,motionConfig,cameraConfig,nisterConfig):
        pass

class motionDataset:
    def __init__(self,outDir,motionConfig,cameraConfig,nisterConfig):
        pass

def getReprojection(currentPoints,currentTriangulated,previousPoints,previousTriangulated,Pl,Pr,scaledH):
    reprojections=[]
    for i in range(0,len(currentPoints)):
        pass
    print(len(currentPoints))


def stereo3DProject(l,r,Q):
    disp=-l[0,0]+r[0,0]
    v=np.ones((4,1),dtype=np.float64)
    v[0,0]=l[0,0]
    v[1,0]=l[1,0]
    v[2,0]=float(disp)
    proj=Q.dot(v)
    return proj/proj[3,0]

def packPoints(stereoLandmarks):
    currentPoints=[]
    previousPoints=[]
    currentLandmarks=[]
    previousLandmarks=[]
    for pointIndex in stereoLandmarks:
        currentPoints.append([pointIndex["Lb"][0,0],pointIndex["Lb"][1,0]])
        currentLandmarks.append(pointIndex["Xb"])
        previousPoints.append([pointIndex["La"][0,0],pointIndex["La"][1,0]])
        previousLandmarks.append(pointIndex["Xa"])   
    return currentPoints,previousPoints,currentLandmarks,previousLandmarks


class idealDataSet:
    def __init__(self,outDir,motionConfig,cameraConfig,nisterConfig):
        self.motion=motionConfig
        self.cameraConfig=cameraConfig
        self.outDir=outDir
        self.nisterConfig=nisterConfig
        self.NisterExtractor=nisterExtract("/media/ryan/EXTRA/output/Simulation",nisterConfig)
        self.pcl=pclExtract("/media/ryan/EXTRA/output/Simulation",nisterConfig)
        self.cvE=cvExtract("/media/ryan/EXTRA/output/Simulation",nisterConfig)
    def generate(self,pointsCurve=[100,250,500,1000,2500],totalH=500):
        totalPoints=max(pointsCurve)
        #pointsCurve.remove(totalPoints)
        for i in range(0,totalH):
            print("H=",i)
            simulationData={}
            simulationData["ID"]=i 
            simulationData["R"]=noisyRotations(self.motion["RotationNoise"])
            simulationData["Tc"]=dominantTranslation(self.motion["TranslationMean"],self.motion["TranslationNoise"])
            simulationData["H"]=createHomog(simulationData["R"]["matrix"],
                                            simulationData["Tc"]["vector"])
            simulationData["Htransform"]=composeTransform(simulationData["R"]["matrix"],
                                            simulationData["Tc"]["vector"])
            simulationData["OC"]={}##operating curve

            for crvTotal in pointsCurve:
                name=str(crvTotal).zfill(5)
                simulationData["OC"][name]={}
                simulationData["OC"][name]["Pts"]=[]
                for pointIndex in range(0,crvTotal):
                    simulationData["OC"][name]["Pts"].append(self.genStereoLandmark(simulationData["Htransform"]))
                    p=simulationData["OC"][name]["Pts"][-1]
                    # print(p["Xa"])
                    # print(p["XaPred"])
                    # print(p["XaQPred"])
                    # print("**************")
                curpts,prvpts,curLand,prevLand=packPoints(simulationData["OC"][name]["Pts"])
                s=self.pcl.rigid_transform_3D(prevLand,curLand)
                c=self.cvE.extractScaledMotion(curpts,curLand,prvpts,prevLand,True)
                simulationData["OC"][name]["nisterResult"]=c
                simulationData["OC"][name]["rigidResult"]=s
                print(getMotion(simulationData["H"]),"ideal")
                # print(getMotion(decomposeTransform(c["H"])),c["nInliers"],"oNister")
                # print(getMotion(decomposeTransform(s["H"])),"rigidBody")
                # print(compareAbsoluteMotion(simulationData["H"],decomposeTransform(s["H"])),"abserr")
                # print(compareAbsoluteMotion(simulationData["H"],decomposeTransform(c["H"])),"abserr2")
                #simulationData["Points"].append(self.genStereoLandmark(simulationData["Htransform"]))###Htransform
            # print(getMotion(simulationData["H"]),"ideal")
            # for curveIndex in pointsCurve:
            #     simulationData["Curves"].append(random.sample(range(0, totalPoints), curveIndex))
            # simulationData["Curves"].append(range(0,totalPoints))##add the 15000 set of indexes
            # print(pointsCurve)
            # for curveArray in simulationData["Curves"]:
            #     currentPoints=[]
            #     previousPoints=[]
            #     currentLandmarks=[]
            #     previousLandmarks=[]
            #     print("OperatingCurveTotal = "+str(len(curveArray)))
            #     for pointIndex in curveArray:
            #         currentPoints.append([simulationData["Points"][pointIndex]["Lb"][0,0],simulationData["Points"][pointIndex]["Lb"][1,0]])
            #         currentLandmarks.append(simulationData["Points"][pointIndex]["Xb"])
            #         previousPoints.append([simulationData["Points"][pointIndex]["La"][0,0],simulationData["Points"][pointIndex]["La"][1,0]])
            #         previousLandmarks.append(simulationData["Points"][pointIndex]["Xa"])
            #     #r=self.NisterExtractor.extractScaledMotion(currentPoints,currentLandmarks,previousPoints,previousLandmarks,True)
            #     s=self.pcl.rigid_transform_3D(previousLandmarks,currentLandmarks)
            #     c=self.cvE.extractScaledMotion(currentPoints,currentLandmarks,previousPoints,previousLandmarks,True)
            #     #simulationData["nisterResult"].append(r)
            #     simulationData["rigidResult"].append(s)
            #     simulationData["nisterResult"].append(c)
            #     print(getMotion(simulationData["H"]),"ideal")
            #     #print("ourNister",r["nInliers"],getMotion(decomposeTransform(r["H"])))
            #     print("opencvNister",c["nInliers"],getMotion(decomposeTransform(c["H"])))
            #     print("rigidBody",getMotion(decomposeTransform(s["H"])))
            # print(len(simulationData["nisterResult"]),len(simulationData["rigidResult"]))
            print("---")
            outFile=self.outDir+"/H_"+str(i).zfill(3)+".p"
            f=open(outFile, 'wb')
            pickle.dump(simulationData,f)
            f.close()
    def genStereoLandmark(self,H):
        validPoint=False
        while(not validPoint):
            simPoint={}
            x=np.random.normal(0,5,1)
            y=np.random.normal(0,5,1)
            z=np.random.normal(0,4,1)
            Point=np.ones((4,1),dtype=np.float64)

            simPoint["Xa"]=copy.deepcopy(Point)
            simPoint["Xa"][0,0]=x
            simPoint["Xa"][1,0]=y
            simPoint["Xa"][2,0]=z
            simPoint["La"]=self.cameraConfig["Pl"].dot(simPoint["Xa"])
            simPoint["La"]=simPoint["La"]/simPoint["La"][2,0]
            simPoint["Ra"]=self.cameraConfig["Pr"].dot(simPoint["Xa"])
            simPoint["Ra"]=simPoint["Ra"]/simPoint["Ra"][2,0]

            simPoint["Xb"]=np.dot(H,simPoint["Xa"])
            simPoint["Xb"]=simPoint["Xb"]/simPoint["Xb"][3,0]
            simPoint["Lb"]=self.cameraConfig["Pl"].dot(simPoint["Xb"])
            simPoint["Lb"]=simPoint["Lb"]/simPoint["Lb"][2,0]
            simPoint["Rb"]=self.cameraConfig["Pr"].dot(simPoint["Xb"])
            simPoint["Rb"]=simPoint["Rb"]/simPoint["Rb"][2,0]
            if(self.withinROI(simPoint["La"])and self.withinROI(simPoint["Lb"])
                and self.withinROI(simPoint["Ra"]) and self.withinROI(simPoint["Rb"])
                and (simPoint["Xa"][2,0]>0) and (simPoint["Xb"][2,0]>0)
                and (simPoint["Xa"][1,0]>-0.5) and (simPoint["Xb"][1,0]>-0.5)):
                validPoint=True
                simPoint["XaPred"]=cv2.triangulatePoints(self.cameraConfig["Pl"],self.cameraConfig["Pr"],
                                    (simPoint["La"][0,0],simPoint["La"][1,0]),
                                    (simPoint["Ra"][0,0],simPoint["Ra"][1,0]))
                simPoint["XaPred"]/=simPoint["XaPred"][3,0]
                simPoint["XbPred"]=cv2.triangulatePoints(self.cameraConfig["Pl"],self.cameraConfig["Pr"],
                                    (simPoint["Lb"][0,0],simPoint["Lb"][1,0]),
                                    (simPoint["Rb"][0,0],simPoint["Rb"][1,0]))
                simPoint["XaQPred"]=stereo3DProject(simPoint["La"],simPoint["Ra"],self.cameraConfig["Q"])
                simPoint["XbQPred"]=stereo3DProject(simPoint["Lb"],simPoint["Rb"],self.cameraConfig["Q"])
        return simPoint
    def withinROI(self,pt):
        if((pt[0]>0)and(pt[0]<self.cameraConfig["width"])):
            if((pt[1]>0)and(pt[1]<self.cameraConfig["height"])):
                return True
            else:
                return False
        else:
            return False

def addOutlier(percentage,inDir,outDir,cameraConfig):
    ####load the files
    ####for i in each operating Curve
    ####calculate the number of outliers
    ####select a random subset of them from the curve
    ####generate a new set of data with nOutliers
    ####store it
    worldFilesSet=os.listdir(inDir)
    for Hpickle in worldFilesSet:
        print(inDir+"/"+Hpickle)
        f=open(inDir+"/"+Hpickle,"r")
        data=pickle.load(f)
        f.close()
        for operatingCurve in data["OC"]:
            print(operatingCurve)
            nOutliers=int(percentage*len(data["OC"][operatingCurve]["Pts"]))
            outliers=random.sample(range(0, len(data["OC"][operatingCurve]["Pts"])),nOutliers)
            data["OC"][operatingCurve]["OutlierIndexes"]=outliers
            print(len(data["OC"][operatingCurve]["OutlierIndexes"]))
            for outlierIndex in outliers:
                Pt=copy.deepcopy(data["OC"][operatingCurve]["Pts"][outlierIndex])
                Pt["La"],Pt["Ra"],Pt["Xa"]=genStereoOutlier(Pt["La"],Pt["Ra"],cameraConfig)
                Pt["Lb"],Pt["Rb"],Pt["Xb"]=genStereoOutlier(Pt["Lb"],Pt["Rb"],cameraConfig)
                data["OC"][operatingCurve]["Pts"][outlierIndex]=Pt
    
        print("written" +outDir+"/"+Hpickle)
        f=open(outDir+"/"+Hpickle,"w")
        pickle.dump(data["OC"],f)
        f.close() 
def genStereoOutlier(l,r,cameraConfig,minNoise=3):
    valid=False
    while(not valid):
        x=np.random.rand(1)*float(cameraConfig["width"])
        y=np.random.rand(1)*float(cameraConfig["height"])#np.random.rand(0,cameraConfig["height"],1)
        pts=np.ones((3,1),dtype=np.float64)
        pts[0,0]=x
        pts[1,0]=y 
        diff=abs(l-pts  ) 
        if((diff[0,0]>minNoise)and(diff[1,0]>minNoise)):
            valid=True
    X=stereo3DProject(pts,r,cameraConfig["Q"])
    rr=cameraConfig["Pr"].dot(X)
    rr/=rr[2,0]
    return pts,rr,X    
def genOutlier(idealPt,cameraConfig,minNoise=3):
    ####generate point from uniform distribution
    ####check if its within largest noise level
    ####
    valid=False
    while(not valid):
        x=np.random.rand(1)*float(cameraConfig["width"])
        y=np.random.rand(1)*float(cameraConfig["height"])#np.random.rand(0,cameraConfig["height"],1)
        pts=np.ones((3,1),dtype=np.float64)
        pts[0,0]=x
        pts[1,0]=y 
        diff=abs(idealPt-pts  ) 
        if((diff[0,0]>minNoise)and(diff[1,0]>minNoise)):
            valid=True

    return pts


def addGaussianNoise(sigma,inDir,outDir,cameraConfig):
    ####for i in each noise level
    ####add noise to the original levels
    ####Re triangulate from the noisy measurements
    ####            -> constrained y noise to -1 and +1
    ####Add it to the Result Object
    ####Store it
    worldFilesSet=os.listdir(inDir)
    for Hpickle in worldFilesSet:
        print(inDir+"/"+Hpickle)
        f=open(inDir+"/"+Hpickle,"r")
        data=pickle.load(f)
        f.close()
        outputData=[]
        for operatingCurve in data["Curves"]:
            newCurve=[]
            for individualIndex in operatingCurve:
                newNoisyPt={}
                newNoisyPt["La"]=addNoise(data["Points"][individualIndex]["La"],sigma,cameraConfig)
                newNoisyPt["Ra"]=addNoise(data["Points"][individualIndex]["Ra"],sigma,cameraConfig)
                newNoisyPt["Lb"]=addNoise(data["Points"][individualIndex]["Lb"],sigma,cameraConfig)
                newNoisyPt["Rb"]=addNoise(data["Points"][individualIndex]["Rb"],sigma,cameraConfig)
                newNoisyPt["Xa"]=cv2.triangulatePoints(cameraConfig["Pl"],cameraConfig["Pr"],
                                            (newNoisyPt["La"][0,0],newNoisyPt["La"][1,0]),
                                            (newNoisyPt["Ra"][0,0],newNoisyPt["Ra"][1,0]))
                newNoisyPt["Xa"]=newNoisyPt["Xa"]/newNoisyPt["Xa"][3,0]
                newNoisyPt["Xb"]=cv2.triangulatePoints(cameraConfig["Pl"],cameraConfig["Pr"],
                            (newNoisyPt["Lb"][0,0],newNoisyPt["Lb"][1,0]),
                            (newNoisyPt["Rb"][0,0],newNoisyPt["Rb"][1,0]))
                newNoisyPt["Xb"]=newNoisyPt["Xb"]/newNoisyPt["Xb"][3,0]
                newCurve.append(newNoisyPt)
            outputData.append(newCurve)
        print("written" +outDir+"/"+Hpickle)
        f=open(outDir+"/"+Hpickle,"w")
        pickle.dump(outputData,f)
        f.close() 
       


def addNoise(idealPt,sigma,cameraConfig):
    valid=False
    while(not valid):
        valid=True
        nx=np.random.normal(0,sigma,1)
        ny=np.random.normal(0,1,1)
        newPt=copy.deepcopy(idealPt)
        newPt[0,0]=newPt[0,0]+nx
        newPt[1,0]=newPt[1,0]+ny

        if((newPt[0,0]>0)and(newPt[0,0]<cameraConfig["width"])):
            if((newPt[1,0]>0)and(newPt[1,0]<cameraConfig["height"])and(abs(ny)<1)):
                valid=True
    return newPt
        #n=np.random.normal(0,noiseLevel,1)
    # def PointCloudMotion(self,simulationData):
    #     ##find centroid
    #     Acentroid=np.zeros((4,1),dtype=np.float64)
    #     Bcentroid=np.zeros((4,1),dtype=np.float64)
    #     for i in simulationData["Points"]:
    #         Acentroid+=i["Xa"]
    #         Bcentroid+=i["Xb"]
    #     Acentroid=(Acentroid/len(simulationData["Points"]))[0:3,0].reshape((3,1))
    #     Bcentroid=(Bcentroid/len(simulationData["Points"]))[0:3,0].reshape((3,1))

    #     H=np.zeros(3,dtype=np.float64)

    #     for i in simulationData["Points"]:    
    #         H = H + ((i["Xa"][0:3,0].reshape((3,1))-Acentroid).dot(
    #                     np.transpose(i["Xb"][0:3,0].reshape((3,1))-Bcentroid)))
    #     u,s,v=np.linalg.svd(H)
    #     R=np.transpose(v).dot(np.transpose(u))
    #     if(np.linalg.det(R)<0):
    #         R[0:3,2]==R[0:3,2]*-1.0

    #     T=R.dot(Acentroid)+Bcentroid
    #     return createHomog(R,T)



# class simulatedCamera:
#     def __init__(self,lInfo,rInfo,Q):
#         self.Pl=np.zeros((3,4),dtype=np.float64)
#         self.Pr=np.zeros((3,4),dtype=np.float64)
#         for row in range(0,3):
#             for col in range(0,4):
#                 self.Pl[row,col]=lInfo.P[row*4 +col]
#                 self.Pr[row,col]=rInfo.P[row*4 +col]
#         self.width=lInfo.width
#         self.height=lInfo.height
#         self.Q=Q
#     def getStereoLandmark(self):
#         validPoint=False
#         while(not validPoint):
#             ustd=0.5*self.width/2.0 ##std deviation
#             vstd=0.5*self.height/2.0
#             leftPt=(np.random.normal(self.width/2.0,ustd,1),
#                     np.random.normal(self.height/2.0,vstd,1))
#             disparity=abs(np.random.normal(0,self.width,1))
#             rightPt=(leftPt[0]+disparity,leftPt[1])
#             vc=np.ones((4,1),dtype=np.float64)
#             vc[0,0]=leftPt[0]
#             vc[1,0]=leftPt[1]
#             vc[2,0]=disparity
#             currentPt=self.Q.dot(vc)
#             if(withinROI(leftPt,self.width,self.height)and
#                    withinROI(rightPt,self.width,self.height)and
#                    (currentPt[2,0]>0)):
#                    validPoint=True
#         simPoint={}
#         simPoint["La"]=leftPt
#         simPoint["Ra"]=rightPt
#         simPoint["Disparity"]=disparity
#         simPoint["Xa"]=currentPt/currentPt[3,0]
#         return simPoint
#     def getIdealSimulation(self,totalPoints,H):
#         ############################
#         ###idealPoint
#         idealPoints=[]
#         for i in range(0,totalPoints):
#             validPoint=False
#             while(not validPoint):
#                 ustd=0.5*self.width/2.0 ##std deviation
#                 vstd=0.5*self.height/2.0
#                 leftPt=(np.random.normal(self.width/2.0,ustd,1),
#                         np.random.normal(self.height/2.0,vstd,1))
#                 disparity=abs(np.random.normal(0,self.width,1))
#                 rightPt=(leftPt[0]+disparity,leftPt[1])
#                 vc=np.ones((4,1),dtype=np.float64)
#                 vc[0,0]=leftPt[0]
#                 vc[1,0]=leftPt[1]
#                 vc[2,0]=disparity
#                 currentPt=self.Q.dot(vc)
#                 nextPt=H.dot(currentPt)
#                 projectedL=self.Pl.dot(nextPt)
#                 projL=(projectedL[0,0]/projectedL[2,0],
#                        projectedL[1,0]/projectedL[2,0])
#                 projectedR=self.Pr.dot(nextPt)
#                 projR=(projectedR[0,0]/projectedR[2,0],
#                        projectedR[1,0]/projectedR[2,0])
#                 if(withinROI(leftPt,self.width,self.height)and
#                    withinROI(rightPt,self.width,self.height)and
#                    withinROI(projL,self.width,self.height)and
#                    withinROI(projR,self.width,self.height)):
#                    simPoint={}
#                    simPoint["La"]=leftPt
#                    simPoint["Ra"]=rightPt
#                    simPoint["Disparity"]=disparity
#                    simPoint["Xa"]=currentPt
#                    simPoint["Xb"]=nextPt
#                    simPoint["Lb"]=projL
#                    simPoint["Rb"]=projR
#                    idealPoints.append(simPoint)
#                    validPoint=True
#         return idealPoints
#     def getBlankImage(self):
#         return np.zeros((self.height,self.width,3),dtype=np.uint8)
#     def estimateMotion(self,simPoints):
#         f=self.Pl[0,0]
#         pp=(self.Pl[0:2,2][0],self.Pl[0:2,2][1])
#         k=self.Pl[0:3,0:3]
#         newPts=np.zeros((len(simPoints),2),dtype=np.float64)
#         oldPts=np.zeros((len(simPoints),2),dtype=np.float64)
#         for i in range(0,len(simPoints)):
#             newPts[i,0]=simPoints[i]["Lb"][0]
#             newPts[i,1]=simPoints[i]["Lb"][1]
#             oldPts[i,0]=simPoints[i]["La"][0]
#             oldPts[i,1]=simPoints[i]["La"][1]
#         E,mask=cv2.findEssentialMat(newPts,oldPts,f,pp,threshold=1)    
#         nInliers,R,T,matchMask=cv2.recoverPose(E,newPts,oldPts,k,mask)
#         print(nInliers)
#         print("----")
#     def reprojectionError(self,inPoint,H):
#         predictedLa=self.Pl.dot(inPoint["Xb"])
#         return predictedLa/predictedLa[2,0],inPoint["La"]
