import cv2
import math
import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_from_euler, quaternion_matrix
from front_end.motion import *
import matplotlib.pyplot as plt
import copy

def noisyRotations(noise=5):
    frame={}
    frame["Roll"]=np.random.normal(0,noise,1)
    frame["Pitch"]=np.random.normal(0,noise,1)
    frame["Yaw"]=np.random.normal(0,noise,1)
    q=quaternion_from_euler(math.radians(frame["Roll"]),
                            math.radians(frame["Pitch"]),
                            math.radians(frame["Yaw"]),'szxy')
    frame["R"]=quaternion_matrix(q)[0:3,0:3]    
    return frame

def dominantTranslation(zBase=0.2,noise=0.1):
    frame={}
    frame["X"]=np.random.normal(0,noise,1)
    frame["Y"]=np.random.normal(0,noise,1)
    frame["Z"]=abs(np.random.normal(zBase,noise,1))
    t=np.zeros((3,1),dtype=np.float64)
    t[0,0]=frame["X"]
    t[1,0]=frame["Y"]
    t[2,0]=frame["Z"]
    frame["T"]=t
    return frame

# def withinROI(pt,width,height):
#     if((pt[0]>0)and(pt[0]<width)):
#         if((pt[1]>0)and(pt[1]<height)):
#             return True
#         else:
#             return False
#     else:
#         return False
# def genIdealPoint(width,height):
#     withinImage=False
#     ustd=0.5*width/2.0 ##std deviation
#     vstd=0.5*height/2.0
#     while(not withinImage):
#         leftPt=(np.random.normal(width/2.0,ustd,1),
#                 np.random.normal(height/2.0,vstd,1))
#         time.sleep(0.1)
#         withinImage=withinROI(leftPt,width,height)        
#     return leftPt   

# class simulatedTrajectory:
#     def __init__(self,total):
#         self.R=[]
#         self.T=[]
#         self.H=[]
#         for i in range(0,total):
#             self.T.append(dominantTranslation())
#             self.R.append(noisyRotations())
#             self.H.append(createHomog(self.R[-1]["R"],
#                                       self.T[-1]["T"]))

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


class simulationDataSet:
    def __init__(self,configDictionary):
        self.settings=configDictionary
        self.slow=[]
        self.medium=[]
        self.fast=[]

    def generateData(self):
        f=self.settings["Pl"][0,0]
        pp=(self.settings["Pl"][0:2,2][0],self.settings["Pl"][0:2,2][1])
        k=np.eye(3,dtype=np.float64)#self.settings["Pl"][0:3,0:3]
        print(f)
        print(pp)
        print(k)
        ####gen slow Data
        for i in range(0,self.settings["TotalSimulations"]):
            simulationData={}
            simulationData["ID"]=i 
            simulationData["Motion"]={}
            simulationData["Motion"]["Results"]={}
            simulationData["Motion"]["Rotation"]=noisyRotations(self.settings["RotationNoise"])
            simulationData["Motion"]["Translation"]=dominantTranslation(self.settings["SlowDeviation"],
                                self.settings["TranslationNoise"])
            simulationData["Motion"]["H"]=createHomog(simulationData["Motion"]["Rotation"]["R"],
                                    simulationData["Motion"]["Translation"]["T"])
            simulationData["Points"]=[]
            newPts=np.zeros((self.settings["TotalPoints"],2),dtype=np.float64)
            oldPts=np.zeros((self.settings["TotalPoints"],2),dtype=np.float64)

		# int totalAverageSamples=0;
		# cv::Mat average=cv::Mat::zeros(3,1,CV_64F);//correctlyScaledTransform
		# cv::Mat K=P(cv::Rect(0,0,3,3));

		# for(int index=0;index<latestInter.currentInlierIndexes.size();index++)
		# {

		# 	if(motionInlierMask.at<bool>(0,index))
		# 	{
		# 		cv::Mat xnew,xold;
		# 		//compute scale from projection 
		# 		//projection pixel in previous frame
		# 		xold=cv::Mat(3,1,CV_64F);
		# 		xold.at<double>(0,0)=previous.inliers.at(latestInter.previousInlierIndexes.at(index)).distance.x;
		# 		xold.at<double>(1,0)=previous.inliers.at(latestInter.previousInlierIndexes.at(index)).distance.y;
		# 		xold.at<double>(2,0)=previous.inliers.at(latestInter.previousInlierIndexes.at(index)).distance.z;

		# 		xnew=cv::Mat(3,1,CV_64F);
		# 		xnew.at<double>(0,0)=current.inliers.at(latestInter.currentInlierIndexes.at(index)).distance.x;
		# 		xnew.at<double>(1,0)=current.inliers.at(latestInter.currentInlierIndexes.at(index)).distance.y;
		# 		xnew.at<double>(2,0)=current.inliers.at(latestInter.currentInlierIndexes.at(index)).distance.z;
		# 		average+=((K.inv()*xold-outR*xnew)*outT.inv(cv::DECOMP_SVD))*outT;
		# 		totalAverageSamples++;
		# 		if(totalAverageSamples==15)
		# 		{
		# 			index=latestInter.currentInlierIndexes.size();
		# 		}
		# 	}
		# }


            for j in range(0,self.settings["TotalPoints"]):
                pt=self.genStereoLandmark(simulationData["Motion"]["H"])
                newPts[j,0]=pt["Lb"][0]
                newPts[j,1]=pt["Lb"][1]
                oldPts[j,0]=pt["La"][0]
                oldPts[j,1]=pt["La"][1]
                simulationData["Points"].append(pt)
            E,mask=cv2.findEssentialMat(newPts,oldPts,f,pp)#,threshold=1)    #
            nInliers,R,T,matchMask=cv2.recoverPose(E,newPts,oldPts,k,mask)

            simulationData["Motion"]["Results"]["Nister"]=nInliers,R,T,matchMask,E
            averageScale=np.zeros((3,3),dtype=np.float64)
            for i in simulationData["Points"]:
                scale=(i["Xa"][0:3,0]-R.dot(i["Xb"][0:3,0])).reshape(3,1).dot(np.transpose(T.reshape(3,1))).dot(np.linalg.pinv(T.dot(np.transpose(T))))
                print(scale)
                averageScale+=scale 
            averageScale=averageScale/len(simulationData["Points"])

                #print(scale)
                #print((i["Xa"][0:3,0]-R.dot(i["Xb"][0:3,0])).shape)
                #print(np.linalg.pinv(T.reshape(3,1).shape))
                #Results=(i["Xa"][0:3,0]-R.dot(i["Xb"][0:3,0])).dot(np.linalg.pinv(T.reshape(3,1)))
                #print("--")#Results)
                # disparity=i["La"][0,0]-i["Ra"][0,0]
                # vc=np.ones((4,1),dtype=np.float64)
                # vc[0,0]=i["La"][0,0]
                # vc[1,0]=i["La"][0,0]
                # vc[2,0]=disparity
                # currentPt=self.settings["Q"].dot(vc)
                # currentPt=currentPt/currentPt[3,0]
                # print(i["Xa"]-currentPt)
                ####get triangulated features from Q


                # print(simulationData["Points"][i]["Xa"][0:3,0])
                # print(simulation)
                # result=(dot(simulationData["Points"][i]["Xa"][0:3,0])-
                #      R.dot(simulationData["Points"][i]["Xb"][0:3,0]))
                # result=result.dot(np.linalg.inv(T))
                # result=result.dot(T)
                # print(result)


# average+=((K.inv()*xold-outR*xnew)*outT.inv(cv::DECOMP_SVD))*outT;

            simulationData["Motion"]["Results"]["PointCloud"]=self.PointCloudMotion(simulationData)
            print("----")
            self.slow.append(simulationData)
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
            simPoint["La"]=self.settings["Pl"].dot(simPoint["Xa"])
            simPoint["La"]=simPoint["La"]/simPoint["La"][2,0]
            simPoint["Ra"]=self.settings["Pr"].dot(simPoint["Xa"])
            simPoint["Ra"]=simPoint["Ra"]/simPoint["Ra"][2,0]

            simPoint["Xb"]=H.dot(simPoint["Xa"])
            simPoint["Xb"]=simPoint["Xb"]/simPoint["Xb"][3,0]
            simPoint["Lb"]=self.settings["Pl"].dot(simPoint["Xb"])
            simPoint["Lb"]=simPoint["Lb"]/simPoint["Lb"][2,0]
            simPoint["Rb"]=self.settings["Pr"].dot(simPoint["Xb"])
            simPoint["Rb"]=simPoint["Rb"]/simPoint["Rb"][2,0]
            if(self.withinROI(simPoint["La"])and self.withinROI(simPoint["Lb"])
                and self.withinROI(simPoint["Ra"]) and self.withinROI(simPoint["Rb"])
                and simPoint["Xa"][2,0]>0 and simPoint["Xb"][2,0]>0):
                validPoint=True
        return simPoint
    def getStereoLandmark(self,H):
        validPoint=False
        while(not validPoint):
            ustd=self.settings["width"]##std deviation
            vstd=self.settings["height"]
            leftPt=(np.random.normal(self.settings["width"]/2.0,ustd,1),
                    np.random.normal(self.settings["height"]/2.0,vstd,1))
            disparity=np.random.normal(0,2.0*self.settings["width"],1)
            rightPt=(leftPt[0]+disparity,leftPt[1])
            vc=np.ones((4,1),dtype=np.float64)
            vc[0,0]=leftPt[0]
            vc[1,0]=leftPt[1]
            vc[2,0]=disparity
            currentPt=self.settings["Q"].dot(vc)
            currentPt=currentPt/currentPt[3,0]
            if(self.withinROI(leftPt)and self.withinROI(rightPt)and(currentPt[2,0]>0)):
                    simPoint={}
                    simPoint["La"]=leftPt
                    simPoint["Ra"]=rightPt
                    simPoint["Disparity"]=disparity
                    simPoint["Xa"]=currentPt
                    validPoint=self.getStereoLandmarkProjection(H,simPoint)
        return simPoint
    def getStereoLandmarkProjection(self,H,simPoint):
        simPoint["Xb"]=H.dot(simPoint["Xa"])
        simPoint["Xb"]=simPoint["Xb"]/simPoint["Xb"][3,0]
        simPoint["Lb"]=self.settings["Pl"].dot(simPoint["Xb"])
        simPoint["Lb"]=simPoint["Lb"]/simPoint["Lb"][2,0]
        simPoint["Rb"]=self.settings["Pr"].dot(simPoint["Xb"])
        simPoint["Rb"]=simPoint["Rb"]/simPoint["Rb"][2,0]
        lbEstimated=self.settings["Pl"].dot(H.dot(simPoint["Xa"]))
        lbEstimated=lbEstimated/lbEstimated[2,0]
        simPoint["LbPredicted"]=lbEstimated
        rbEstimated=self.settings["Pr"].dot(H.dot(simPoint["Xa"]))
        rbEstimated=rbEstimated/rbEstimated[2,0]
        simPoint["RbPredicted"]=rbEstimated
        if(self.withinROI(simPoint["Lb"])and self.withinROI(simPoint["Rb"])and(simPoint["Xb"][2,0]>0)):
            return True
        else:
            return False
    def withinROI(self,pt):
        if((pt[0]>0)and(pt[0]<self.settings["width"])):
            if((pt[1]>0)and(pt[1]<self.settings["height"])):
                return True
            else:
                return False
        else:
            return False 
    def PointCloudMotion(self,simulationData):
        ##find centroid
        Acentroid=np.zeros((4,1),dtype=np.float64)
        Bcentroid=np.zeros((4,1),dtype=np.float64)
        for i in simulationData["Points"]:
            Acentroid+=i["Xa"]
            Bcentroid+=i["Xb"]
        Acentroid=(Acentroid/len(simulationData["Points"]))[0:3,0].reshape((3,1))
        Bcentroid=(Bcentroid/len(simulationData["Points"]))[0:3,0].reshape((3,1))

        H=np.zeros(3,dtype=np.float64)

        for i in simulationData["Points"]:    
            H = H + ((i["Xa"][0:3,0].reshape((3,1))-Acentroid).dot(
                        np.transpose(i["Xb"][0:3,0].reshape((3,1))-Bcentroid)))
        u,s,v=np.linalg.svd(H)
        R=np.transpose(v).dot(np.transpose(u))
        if(np.linalg.det(R)<0):
            R[0:3,2]==R[0:3,2]*-1.0

        T=R.dot(Acentroid)+Bcentroid
        return createHomog(R,T)



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
