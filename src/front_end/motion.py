import numpy as np
from tf.transformations import quaternion_from_euler,quaternion_matrix,euler_from_matrix
from math import pi,radians,degrees
import rosbag
import time
import cv2
import copy
from scipy.optimize import least_squares
import decimal

def composeR(roll,pitch,yaw,degrees=True,dict=True):
    if(degrees):
        q=quaternion_from_euler(radians(roll),
                                radians(pitch),
                                radians(yaw),'szxy')
    else:
        q=quaternion_from_euler(roll,
                                pitch,
                                yaw,'szxy')     
    return quaternion_matrix(q)[0:3,0:3]  

    # q=quaternion_from_euler(math.radians(frame["Roll"]),
    #                         math.radians(frame["Pitch"]),
    #                         math.radians(frame["Yaw"]),'szxy')
    # frame["matrix"]=quaternion_matrix(q)[0:3,0:3]  

def getDNister():
    out=np.zeros((3,3),dtype=np.float64)
    out[0,1]=1
    out[1,0]=-1
    out[2,2]=1
    return out
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



     # if(nisterResults["nInliers"]>0):
        #     averageScale=np.zeros((3,3),dtype=np.float64)
        #     countedIn=0
        #     Tinv=getPsudeoInverseColumn(T)
        #     T3x3=get3x3Translation(T)
        #     for index in range(0,len(currentPoints)):
        #         if(nisterResults["inlierMask"][index,0]>0):
        #             Xb=currentTriangulated[index][0:3,0].reshape(3,1)
        #             Xa=previousTriangulated[index][0:3,0].reshape(3,1)
        #             scale=(Xb-R.dot(Xa)).dot(np.transpose(T.reshape(3,1))).dot(Tinv)
        #             averageScale+=scale 
        #             countedIn+=1
        #     averageScale=averageScale/nisterResults["nInliers"]
        #     newT=averageScale.dot(T)
        #     nisterResults["T"]=newT
        #     nisterResults["R"]=R
        #     nisterResults["H"]=createHomog(nisterResults["R"],nisterResults["T"])  


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

def getRotationMotion(R):
    Result={}
    angles=copy.deepcopy(euler_from_matrix(R,'szxy'))
    Result["Roll"]=57.2958*angles[0]
    Result["Pitch"]=57.2958*angles[1]
    Result["Yaw"]=57.2958*angles[2]
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

def compareAbsoluteMotion(H,Hest):
    Hstruct=getMotion(H)
    HestStruct=getMotion(Hest)
    Result={}
    Result["Roll"]=round(getAbsoluteError(Hstruct["Roll"],HestStruct["Roll"]),3)
    Result["Pitch"]=round(getAbsoluteError(Hstruct["Pitch"],HestStruct["Pitch"]),3)
    Result["Yaw"]=round(getAbsoluteError(Hstruct["Yaw"],HestStruct["Yaw"]),3)
    Result["X"]=round(getAbsoluteError(Hstruct["X"],HestStruct["X"])*1000,3)###in mm
    Result["Y"]=round(getAbsoluteError(Hstruct["Y"],HestStruct["Y"])*1000,3)
    Result["Z"]=round(getAbsoluteError(Hstruct["Z"],HestStruct["Z"])*1000,3)
    return Result   

def compareUnitMotion(H,Hest):
    ###normalize the original to unit
    Hstruct=getMotion(H)
    HestStruct=getMotion(Hest)  

    structNorm=np.sqrt(Hstruct["X"]**2 + Hstruct["Y"]**2 + Hstruct["Z"]**2)
    Hstruct["X"]/=structNorm
    Hstruct["Y"]/=structNorm
    Hstruct["Z"]/=structNorm
    Result={}
    Result["Roll"]=getPercentError(Hstruct["Roll"],HestStruct["Roll"])
    Result["Pitch"]=getPercentError(Hstruct["Pitch"],HestStruct["Pitch"])
    Result["Yaw"]=getPercentError(Hstruct["Yaw"],HestStruct["Yaw"])
    Result["X"]=getPercentError(Hstruct["X"],HestStruct["X"])
    Result["Y"]=getPercentError(Hstruct["Y"],HestStruct["Y"])
    Result["Z"]=getPercentError(Hstruct["Z"],HestStruct["Z"])
    print(np.sqrt(Hstruct["X"]**2+Hstruct["Y"]**2+Hstruct["Z"]),
            np.sqrt(HestStruct["X"]**2+HestStruct["Y"]**2+HestStruct["Z"]))
    return Result

def getUnitTranslation(H):
    T=copy.deepcopy(H[0:3,3])
    return T


def getAbsoluteError(ideal,measured):
    diff=abs(ideal-measured)
    return diff

def getPercentError(ideal,measured):
    diff=ideal-measured
    pError=100*diff/abs(float(ideal))
    return pError

def getPsuedoInverseColumn(T):
    return np.linalg.pinv(get3x3Translation(T))

def get3x3Translation(T):
    return T.dot(np.transpose(T.reshape(3,1)))

def estimateScale(xPrev,xCurrent,R,T,inliers):
    Rinv=R
    Ti=T
    averageScale=np.zeros((3,3),dtype=np.float64)
    countedIn=0
    Tinv=getPsuedoInverseColumn(Ti)
    #print("counted",list(inliers).count(255))
    for index in range(0,len(xCurrent)):
        if(inliers[index,0]>0):
            Xb=xCurrent[index][0:3,0].reshape(3,1)
            Xa=xPrev[index][0:3,0].reshape(3,1)
            scale=(Xb-R.dot(Xa)).dot(np.transpose(Ti.reshape(3,1))).dot(Tinv)
            averageScale+=scale 
            countedIn+=1
    averageScale=averageScale/countedIn
    T=averageScale.dot(Ti)
    return averageScale,T,countedIn

# def getPCLmotion(currentTriangulated,previousTriangulated):
#     ##find centroid
#     Acentroid=np.zeros((4,1),dtype=np.float64)
#     Bcentroid=np.zeros((4,1),dtype=np.float64)
#     for i in previousTriangulated:
#         Acentroid+=i
#     for i in currentTriangulated:
#         Bcentroid+=i
#     Acentroid=(Acentroid/len(previousTriangulated))[0:3,0].reshape((3,1))
#     Bcentroid=(Bcentroid/len(currentTriangulated))[0:3,0].reshape((3,1))

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



    # def PointCloudMotion(self,simulationData):
    #     ##find centroid
    #     Acentroid=np.zeros((4,1),dtype=np.float64)
    #     Bcentroid=np.zeros((4,1),dtype=np.float64)
    #     for i in simulationData["Points"]:
    #         Acentroid+=i["Xa"]
    #         Bcentroid+=i["Xb"]
    #     Acentroid=(Acentroid/len(simulationData["Points"]))[0:3,0].reshape((3,1))
    #     Bcentroid=(Bcentroid/len(simulationData["Points"]))[0:3,0].reshape((3,1))


class pclExtract:
    def __init__(self,rootDir,extractConfig):
        self.root=rootDir
        self.output=rootDir+"/pcl"
        self.extract=extractConfig
    def closedForm(self,currentTriangulated,previousTriangulated):
        ##find centroid
        Acentroid=np.zeros((4,1),dtype=np.float64)
        Bcentroid=np.zeros((4,1),dtype=np.float64)
        for i in previousTriangulated:
            Acentroid+=i
        for i in currentTriangulated:
            Bcentroid+=i
        Acentroid=(Acentroid/len(previousTriangulated))[0:3,0].reshape((3,1))
        Bcentroid=(Bcentroid/len(currentTriangulated))[0:3,0].reshape((3,1))
        H=np.zeros(3,dtype=np.float64)

        for i in range(0,len(currentTriangulated)):    
            H = H + ((previousTriangulated[i][0:3,0].reshape((3,1))-Acentroid).dot(
                        np.transpose(currentTriangulated[i][0:3,0].reshape((3,1))-Bcentroid)))
        H/=float(len(currentTriangulated))

        u,s,v=np.linalg.svd(H,True,True)
        #print('u',u)
        #print('s',s)
        #print(s.shape)
        #print('v',v)
        #####
        checkCount=np.linalg.det(u)*np.linalg.det(np.transpose(v))
        s=np.eye(3,dtype=np.float64)
        if(checkCount==1):
            s=np.eye(3,dtype=np.float64)
        else:
            s[-1,-1]=-1
        print('s2',s)
        R=u.dot(s).dot(np.transpose(v))
        T=-R.dot(Acentroid)+Bcentroid
        return createHomog(R,T)
    def rigid_transform_3D(self,previousLandmarks, currentLandmarks):
        #assert len(A) == len(B)
        n=len(previousLandmarks)
        A=np.mat(np.random.rand(n,3),dtype=np.float64)
        B=np.mat(np.random.rand(n,3),dtype=np.float64)
        for a in range(0,len(currentLandmarks)):
            A[a,0]=previousLandmarks[a][0,0]
            A[a,1]=previousLandmarks[a][1,0]
            A[a,2]=previousLandmarks[a][2,0]
            B[a,0]=currentLandmarks[a][0,0]
            B[a,1]=currentLandmarks[a][1,0]
            B[a,2]=currentLandmarks[a][2,0]

        N = A.shape[0]; # total points

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        #print(centroid_A)
        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))
        #print(AA.shape)
        #print(BB.shape)
        #print(np.transpose(AA).shape)
        # dot is matrix multiplication for array
        H = np.transpose(AA).dot(BB)
        #print(H.shape)
        U, S, Vt = np.linalg.svd(H)

        R = Vt.T * U.T

        # special reflection case
        if(np.linalg.det(R) < 0):
            #print "Reflection detected"
            Vt[2,:] *= -1
            R = Vt.T * U.T
        #print(R.shape,centroid_A.T.shape,centroid_B.T.shape)
        #print((-R*centroid_A.T).shape)
        t = -R.dot(centroid_A.T) + centroid_B.T
        #print(R,t)
        #print(t.shape)
        #print t
        out={}
        out["R"]=R
        out["T"]=t
        out["H"]=createHomog(R, t)
        return out

class cvExtract:
    def __init__(self,rootDir,extractConfig):
        self.root=rootDir
        self.output=rootDir +"/cv"
        self.extract=extractConfig
    def extractMotion(self,currentPoints,previousPoints,dictionary=False):
        newPts=np.zeros((len(currentPoints),2),dtype=np.float64)
        oldPts=np.zeros((len(currentPoints),2),dtype=np.float64)
        for j in range(0,len(currentPoints)):

            newPts[j,0]=currentPoints[j][0]#simulationPoints[j]["Lb"][0]
            newPts[j,1]=currentPoints[j][1]#simulationPoints[j]["Lb"][1]
            oldPts[j,0]=previousPoints[j][0]#simulationPoints[j]["La"][0]
            oldPts[j,1]=previousPoints[j][1]#simulationPoints[j]["La"][1]      
        E,mask=cv2.findEssentialMat(oldPts,newPts,self.extract["f"],self.extract["pp"],method=cv2.RANSAC,threshold=self.extract["threshold"],prob=self.extract["probability"])
        nInliers=list(mask).count(1)
        #print(nInliers,"first")
        if(not dictionary):
            H=createHomog(R,T)
            return E,nInliers,mask
        else:
            Results={}
            Results["inlierMask"]=mask
            Results["nInliers"]=nInliers
            Results["inlierRatio"]=nInliers/float(len(currentPoints))
            Results["E"]=E
            return Results  
    def extractScaledMotion(self,currentPoints,currentTriangulated,previousPoints,previousTriangulated,dictionary=False):
        newPts=np.zeros((len(currentPoints),2),dtype=np.float64)
        oldPts=np.zeros((len(currentPoints),2),dtype=np.float64)
        for j in range(0,len(currentPoints)):

            newPts[j,0]=currentPoints[j][0]
            newPts[j,1]=currentPoints[j][1]
            oldPts[j,0]=previousPoints[j][0]
            oldPts[j,1]=previousPoints[j][1]      
        nisterResults=self.extractMotion(currentPoints,previousPoints,True)
        R1,R2,t=cv2.decomposeEssentialMat(nisterResults["E"])
        nInliers,R,T,matchMask=cv2.recoverPose(nisterResults["E"],oldPts,newPts,self.extract["k"])
        nisterResults["inlierMask"]=matchMask
        nisterResults["nInliers"]=nInliers
        #print(nInliers)
        nisterResults["T"]=T
        nisterResults["R"]=R
        nisterResults["H"]=createHomog(R,T)
        if(nisterResults["nInliers"]>0):
            s,t,inl=estimateScale(previousTriangulated,currentTriangulated,R,T,matchMask)
            nisterResults["T"]=t
            nisterResults["nInliers"]=inl
        nisterResults["H"]=createHomog(nisterResults["R"],nisterResults["T"])
        if(not dictionary):
            return nisterResults["H"]
        else:
            return nisterResults


# class simpleBundle(self,rootDir,extractConfig):
#         self.root=rootDir
#         self.output=rootDir+"/Nister"
#         self.extract=extractConfig
#     def frameBundle(self,currentPoints,currentTriangulated,previousPoints,previousTriangulated,dictionary=False):
#     return 0

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
        E,mask=cv2.findEssentialMat(oldPts,newPts,self.extract["f"],self.extract["pp"],method=cv2.RANSAC,threshold=self.extract["threshold"],prob=self.extract["probability"])
        nInliers=list(mask).count(1)
        if(not dictionary):
            H=createHomog(R,T)
            return E,nInliers,mask
        else:
            Results={}
            Results["inlierMask"]=mask
            Results["nInliers"]=nInliers
            Results["inlierRatio"]=nInliers/float(len(currentPoints))
            Results["E"]=E
            return Results          
    def extractScaledMotion(self,currentPoints,currentTriangulated,previousPoints,previousTriangulated,dictionary=False):
        newPts=np.zeros((len(currentPoints),2),dtype=np.float64)
        oldPts=np.zeros((len(currentPoints),2),dtype=np.float64)
        for j in range(0,len(currentPoints)):

            newPts[j,0]=currentPoints[j][0]
            newPts[j,1]=currentPoints[j][1]
            oldPts[j,0]=previousPoints[j][0]
            oldPts[j,1]=previousPoints[j][1]      
        nisterResults=self.extractMotion(currentPoints,previousPoints,True)
        decomp=self.essential(nisterResults["E"])
        R1,R2,t=decomp["Ra"],decomp["Rb"],decomp["T"]
        h=[createHomog(R1,t),createHomog(R1,-t),createHomog(R2,t),createHomog(R2,-t)]
        #print(getRotationMotion(R1),getRotationMotion(R2))
        P0=self.composeCamera(np.eye(3,dtype=np.float64),np.zeros((3,1),dtype=np.float64))
        Pa=self.composeCamera(R1,t)
        Pb=self.composeCamera(R1,-t)
        Pc =self.composeCamera(R2,t)
        Pd=self.composeCamera(R2,-t)
        votes=[0,0,0,0]
        for i in range(0,len(currentPoints)):
            if(nisterResults["inlierMask"][i,0]>0):
                x=cv2.triangulatePoints(P0,Pa,oldPts[i],newPts[i])
                x/=x[3,0]

                x2=cv2.triangulatePoints(P0,Pb,oldPts[i],newPts[i])
                x2/=x2[3,0]

                x3=cv2.triangulatePoints(P0,Pc,oldPts[i],newPts[i])
                x3/=x3[3,0]
                x4=cv2.triangulatePoints(P0,Pd,oldPts[i],newPts[i])
                x4/=x4[3,0]
                if(x[2,0]>0):
                    if(t[2,0]>0):
                        votes[0]+=1
                if(x2[2,0]>0):
                    if(-t[2,0]>0):
                        votes[1]+=1
                if(x3[2,0]>0):
                    if(t[2,0]>0):
                        votes[2]+=1                    
                if(x4[2,0]>0):
                    if(-t[2,0]>0):
                        votes[3]+=1                    
        print(votes)
        # for i in range(0,len(currentPoints)):
        #     if(nisterResults["inlierMask"][i,0]==1):
        #         x=cv2.triangulatePoints(P0,Pa,oldPts[i],newPts[i])
        #         c1=x[3,0]*x[2,0]
        #         c2=Pa.dot(x)[2,0]*x[3,0]
        #         if((c1>0)and(c2>0)):
        #             votes[0]+=1
        #         elif((c1<0)and(c2<0)):
        #             votes[1]+=1
        #         elif(c1*c2<0):
        #             if(x[2,0]*decomp["Ht"].dot(x)[3,0]>0):
        #                 votes[2]+=1
        #             else:
        #                 votes[3]+=1
        #         else:
        #             print("CUKC UP")
        indexFound=votes.index(max(votes))
        if(indexFound==0):
            R=R1
            T=t
        elif(indexFound==1):
            R=R1
            T=-t
        elif(indexFound==2):
            R=R2
            T=t
        else:
            R=R2
            T=-t    
        nisterResults["nInliers"]=max(votes)       
        nisterResults["R"]=R
        nisterResults["T"]=T
        if(nisterResults["nInliers"]>0):
            s,t,inl=estimateScale(previousTriangulated,currentTriangulated,R,T,nisterResults["inlierMask"])
            nisterResults["T"]=t
            nisterResults["nInliers"]=inl
        nisterResults["H"]=createHomog(nisterResults["R"],nisterResults["T"])
        if(not dictionary):
            return nisterResults["H"]
        else:
            return nisterResults
    def essential(self,inE):
        output={}
        u,s,vT=np.linalg.svd(inE,True,True)

        if(np.linalg.det(u)<0):
            u*=-1
        if(np.linalg.det(vT)<0):
            vT*=-1
        output["Ra"]=u.dot(getDNister()).dot(vT)
        output["Rb"]=u.dot(np.transpose(getDNister())).dot(vT)
        output["T"]=u[0:3,2].reshape(3,1)
        output["Ht"]=np.eye(4,dtype=np.float64)
        output["Ht"][3,3]=-1
        output["Ht"][3,0:3]=-2*vT[0:3,2].reshape(1,3)
        return output
    def composeCamera(self,R,T):
        P=np.zeros((3,4),dtype=np.float64)
        P[0:3,0:3]=R
        P[0:3,3]=T.reshape(3)
        P=self.extract["k"].dot(P)
        return P
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

################################
###RANSAC
################################


def pclRANSAC(previousTriangulated,currentTriangulated,Pl,Pr):
    iterations=50
    reprojThresh=2
    bestfit=None
    besterr=np.inf
    best_inlier_idxs=None
    k=0
    print(previousTriangulated[0])
    while k < iterations:
        k+=1
        ###get random subset
        ###repack them
        a=rigid_transform_3D(previousTriangulated,currentTriangulated)
    return a


def rigid_transform_3D(previousLandmarks, currentLandmarks):
        #assert len(A) == len(B)
        n=len(previousLandmarks)
        A=np.mat(np.random.rand(n,3),dtype=np.float64)
        B=np.mat(np.random.rand(n,3),dtype=np.float64)
        for a in range(0,len(currentLandmarks)):
            A[a,0]=previousLandmarks[a][0,0]
            A[a,1]=previousLandmarks[a][1,0]
            A[a,2]=previousLandmarks[a][2,0]
            B[a,0]=currentLandmarks[a][0,0]
            B[a,1]=currentLandmarks[a][1,0]
            B[a,2]=currentLandmarks[a][2,0]

        N = A.shape[0]; # total points

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        #print(centroid_A)
        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))
        #print(AA.shape)
        #print(BB.shape)
        #print(np.transpose(AA).shape)
        # dot is matrix multiplication for array
        H = np.transpose(AA).dot(BB)
        #print(H.shape)
        U, S, Vt = np.linalg.svd(H)

        R = Vt.T * U.T

        # special reflection case
        if(np.linalg.det(R) < 0):
            #print "Reflection detected"
            Vt[2,:] *= -1
            R = Vt.T * U.T
        #print(R.shape,centroid_A.T.shape,centroid_B.T.shape)
        #print((-R*centroid_A.T).shape)
        t = -R.dot(centroid_A.T) + centroid_B.T
        #print(R,t)
        #print(t.shape)
        #print t
        out={}
        out["R"]=R
        out["T"]=t
        out["H"]=createHomog(R, t)
        return out
# class pclRANSAC:
#     def estimate(currentPoints,previousPoints,currentLandmarks,previousLandmarks):
#         print("AC")


def composeCam(R,T):
    P=np.zeros((3,4),dtype=np.float64)
    P[0:3,0:3]=R
    P[0:3,3]=T 
    P= kSettings["k"].dot(P)
    return P

class BAextractor:
    def __init__(self,cameraSettings):
        self.settings=copy.deepcopy(cameraSettings)
    def extract(self,currentPoints,currentLandmarks,
                previousPoints,previousLandmarks):
        print("--")
        ans=least_squares(self.error,np.array([0,0,0,0,0,0]),verbose=True,max_nfev=500,
                            args=(currentPoints,currentLandmarks,
                        previousPoints,previousLandmarks))
        result={}
        result["Stats"]=ans
        result["T"]=np.zeros((3,1),dtype=np.float64)
        result["T"][0,0]=ans.x[3]
        result["T"][1,0]=ans.x[4]
        result["T"][2,0]=ans.x[5]
        # result["Roll"]=math.degrees(ans.x[0])
        # result["Pitch"]=math.degrees(ans.x[1])
        # result["Yaw"]=math.degrees(ans.x[2])
        result["R"]=composeR(math.degrees(ans.x[0]),
                            math.degrees(ans.x[1]),
                            math.degrees(ans.x[2]))
        result["H"]=createHomog(result["R"],result["T"])
        return result
    def error(self,x, *args, **kwargs):
        q=quaternion_from_euler(x[0],x[1],x[2],'szxy')
        Rest=quaternion_matrix(q)[0:3,0:3]  
        T=x[3:]
        Ht=createHomog(Rest,T)
        totalRMS=0
        index=0
        P=np.zeros((3,4),dtype=np.float64)
        P[0:3,0:3]=Rest
        P[0:3,3]=T 
        Pb= self.settings["k"].dot(P)
        for dataIndex in range(0,len(args[0])):
            newPixel=Pb.dot(args[1][dataIndex])
            newPixel= newPixel/newPixel[2,0]
            e=abs(args[2][dataIndex][0]-newPixel[0,0])+abs(args[2][dataIndex][1]-newPixel[1,0])
            totalRMS+=e
            index+=1
        return totalRMS
