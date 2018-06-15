import numpy as np
from tf.transformations import quaternion_from_euler,quaternion_matrix,euler_from_matrix
from math import pi
import rosbag
import time
import cv2
import copy
def deserialHomography(arrayIn):
    outHomography=np.zeros((4,4),dtype=np.float64)
    row=0
    col=0
    for row in range(0,4):
        for col in range(0,4):
            outHomography[row,col]=arrayIn[row*4+col]
    return outHomography 

def getHomogZeros():
    out=np.zeros((4,1),dtype=np.float64)
    out[3,0]=1
    return out

def createHomog(R=np.eye(3,dtype=np.float64),
                T=np.zeros((3,1),np.float64)):
    output=np.eye(4,dtype=np.float64)
    output[0:3,0:3]=R
    output[0:3,3]=T.reshape(3)
    return output

def composeTransform(R,T):
    ####H=[R -RT]
    ######[0   1]
    return createHomog(R,-R.dot(T))

def decomposeTransform(H):
    R=copy.deepcopy(H[0:3,0:3])
    T=-1*np.linalg.inv(R).dot(H[0:3,3])
    return createHomog(R,T)


def getMotion(H):
    Result={}
    angles=copy.deepcopy(euler_from_matrix(H[0:3,0:3],'szxy'))
    Result["Roll"]=57.2958*angles[0]
    Result["Pitch"]=57.2958*angles[1]
    Result["Yaw"]=57.2958*angles[2]
    Result["X"]=copy.deepcopy(H[0,3])
    Result["Y"]=copy.deepcopy(H[1,3])
    Result["Z"]=copy.deepcopy(H[2,3])
    return Result

def compareMotion(H,Hest):
    Hstruct=getMotion(H)
    HestStruct=getMotion(Hest)
    Result={}
    Result["Roll"]=getPercentError(Hstruct["Roll"],HestStruct["Roll"])
    Result["Pitch"]=getPercentError(Hstruct["Pitch"],HestStruct["Pitch"])
    Result["Yaw"]=getPercentError(Hstruct["Yaw"],HestStruct["Yaw"])
    Result["X"]=getPercentError(Hstruct["X"],HestStruct["X"])
    Result["Y"]=getPercentError(Hstruct["Y"],HestStruct["Y"])
    Result["Z"]=getPercentError(Hstruct["Z"],HestStruct["Z"])
    return Result

def getPercentError(ideal,measured):
    diff=abs(ideal-measured)
    pError=100*diff/abs(float(ideal))
    return pError

def getPsudeoInverseColumn(T):
    return np.linalg.pinv(get3x3Translation(T))

def get3x3Translation(T):
    return T.dot(np.transpose(T.reshape(3,1)))

class nisterExtract:
    def __init__(self,rootDir,extractConfig):
        self.root=rootDir
        self.output=rootDir+"/Nister"
        self.extract=extractConfig
    def extractMotion(self,currentPoints,previousPoints,dictionary=False):
        newPts=np.zeros((len(currentPoints),2),dtype=np.float64)
        oldPts=np.zeros((len(currentPoints),2),dtype=np.float64)
        for j in range(0,len(currentPoints)):
            newPts[j,0]=currentPoints[j][0]#simulationPoints[j]["Lb"][0]
            newPts[j,1]=currentPoints[j][1]#simulationPoints[j]["Lb"][1]
            oldPts[j,0]=previousPoints[j][0]#simulationPoints[j]["La"][0]
            oldPts[j,1]=previousPoints[j][1]#simulationPoints[j]["La"][1]      
        E,mask=cv2.findEssentialMat(oldPts,newPts,self.extract["f"],self.extract["pp"])
        nInliers,R,T,matchMask=cv2.recoverPose(E,oldPts,newPts,self.extract["k"],mask)
        if(not dictionary):
            H=createHomog(R,T)
            return H,E,R,T,matchMask,nInliers
        else:
            Results={}
            Results["H"]=createHomog(R,T)
            Results["T"]=T
            Results["R"]=R
            Results["inlierMask"]=matchMask
            Results["nInliers"]=nInliers
            Results["inlierRatio"]=nInliers/float(len(currentPoints))
            Results["E"]=E
            return Results         
    def extractScaledMotion(self,currentPoints,currentTriangulated,previousPoints,previousTriangulated,dictionary=False):
        nisterResults=self.extractMotion(currentPoints,previousPoints,True)
        averageScale=np.zeros((3,3),dtype=np.float64)
        countedIn=0
        R=nisterResults["R"]
        T=nisterResults["T"]
        Tinv=getPsudeoInverseColumn(T)
        T3x3=get3x3Translation(T)
        for index in range(0,len(currentPoints)):
            if(nisterResults["inlierMask"][index,0]==255):
                Xa=previousTriangulated[index][0:3,0].reshape(3,1)
                Xb=currentTriangulated[index][0:3,0].reshape(3,1)
                scale=(Xb-R.dot(Xa)).dot(np.transpose(T.reshape(3,1))).dot(Tinv)
                averageScale+=scale 
                countedIn+=1
        
        averageScale=averageScale/nisterResults["nInliers"]
        newT=averageScale.dot(nisterResults["T"])
        nisterResults["T"]=newT
        nisterResults["H"]=createHomog(nisterResults["R"],nisterResults["T"])
        if(not dictionary):
            return nisterResults["H"]
        else:
            return nisterResults
    def extractMotionFull(self,currentPoints,PreviousPoints):
        newPts=np.zeros((len(currentPoints),2),dtype=np.float64)

        # curveID=str(len(curve))
        #         HResults[curveID]={}
        #         simulationPoints=[]
        #         for pointIndex in curve:
        #             simulationPoints.append(data["Points"][pointIndex])
        #             newPts=np.zeros((len(simulationPoints),2),dtype=np.float64)
        #             oldPts=np.zeros((len(simulationPoints),2),dtype=np.float64)
        #         for j in range(0,len(simulationPoints)):
        #             newPts[j,0]=simulationPoints[j]["Lb"][0]
        #             newPts[j,1]=simulationPoints[j]["Lb"][1]
        #             oldPts[j,0]=simulationPoints[j]["La"][0]
        #             oldPts[j,1]=simulationPoints[j]["La"][1]
        #         E,mask=cv2.findEssentialMat(newPts,oldPts,self.extract["f"],self.extract["pp"])
        #                                     #,prob=self.extract["probability"],threshold=self.extract["threshold"])#,threshold=1)    #
        #         nInliers,R,T,matchMask=cv2.recoverPose(E,newPts,oldPts,self.extract["k"],mask)
        #         averageScale=np.zeros((3,3),dtype=np.float64)
        #         countedIn=0
        #         for index in range(0,len(simulationPoints)):
        #             i=simulationPoints[index]
        #             if(matchMask[index,0]==255):
        #                 scale=(i["Xa"][0:3,0]-R.dot(i["Xb"][0:3,0])).reshape(3,1).dot(np.transpose(T.reshape(3,1))).dot(np.linalg.pinv(T.dot(np.transpose(T))))
        #                 averageScale+=scale 
        #                 countedIn+=1
        #         averageScale=averageScale/nInliers
        #         T=averageScale.dot(T)  
        #         original=createHomog(R,T)
        #         HResults[curveID]["H"]=np.linalg.inv(original)
        #         print(getMotion(HResults[curveID]["H"]))
        #         HResults[curveID]["Motion"]=getMotion(HResults[curveID]["H"]) 
        #         HResults[curveID]["inlierMask"]=matchMask
        #         HResults[curveID]["nInlier"]=nInliers
        #         HResults[curveID]["inlierRatio"]=nInliers/float(len(simulationPoints))
        #         HResults[curveID]["E"]=E
        #         HResults[curveID]["MotionError"]=compareMotion(HResults[curveID]["H"],data["H"])
        #         HResults[curveID]["CurveID"]=len(simulationPoints)
        #         HResults[curveID]["PointResults"]=[]
        #         #####get reprojection results
        #         for index in range(0,len(simulationPoints)):
        #             i=simulationPoints[index]
        #             if(matchMask[index,0]==255):
        #                 HResults[curveID]["PointResults"].append(self.getLandmarkReprojection(i,HResults[curveID]["H"]) )