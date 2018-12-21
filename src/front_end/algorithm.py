import numpy as np
from cv_bridge import CvBridge

import copy
from scipy.optimize import least_squares
from bumblebee.motion import *
from bumblebee.baseTypes import *
from math import pi,radians,degrees
from bumblebee.stereo import *
from bumblebee.camera import *
import matplotlib.pyplot as plt
import matplotlib.style as sty
import statistics












def rigid_transform_3D(previousLandmarks, currentLandmarks):
    N=previousLandmarks.shape[1]
    centroid_A = np.mean(previousLandmarks.T, axis=0)
    centroid_B = np.mean(currentLandmarks.T, axis=0)

    AA = copy.deepcopy(previousLandmarks.T - np.tile(centroid_A, (N, 1)))
    BB = copy.deepcopy(currentLandmarks.T - np.tile(centroid_B, (N, 1)))
    H = np.transpose(AA).dot(BB)

    U, S, Vt = np.linalg.svd(H)
    R = (Vt.T).dot( U.T)
    # special reflection case
    if(np.linalg.det(R) < 0):
        Vt[2,:] *= -1
        R = (Vt.T).dot(U.T)
    t = -R.dot(centroid_A.T) + centroid_B.T

    return createHomog(R,t)


#     ########
#     ##THE ORIGINAL THAT WORKED
# def rigid_transform2_3D(previousLandmarks, currentLandmarks):

#     n=len(previousLandmarks)
#     A=np.mat(np.random.rand(n,3),dtype=np.float64)
#     B=np.mat(np.random.rand(n,3),dtype=np.float64)
#     for a in range(0,len(currentLandmarks)):
#         A[a,0]=previousLandmarks[a].X[0,0]
#         A[a,1]=previousLandmarks[a].X[1,0]
#         A[a,2]=previousLandmarks[a].X[2,0]
#         B[a,0]=currentLandmarks[a].X[0,0]
#         B[a,1]=currentLandmarks[a].X[1,0]
#         B[a,2]=currentLandmarks[a].X[2,0]
#     N = A.shape[0]; # total points

#     centroid_A = np.mean(A, axis=0)
#     centroid_B = np.mean(B, axis=0)
    
#     # centre the points
#     AA = A - np.tile(centroid_A, (N, 1))
#     BB = B - np.tile(centroid_B, (N, 1))

#     # dot is matrix multiplication for array
#     H = np.transpose(AA).dot(BB)

#     U, S, Vt = np.linalg.svd(H)

#     R = Vt.T * U.T

#     # special reflection case
#     if(np.linalg.det(R) < 0):
#         Vt[2,:] *= -1
#         R = Vt.T * U.T

#     t = -R.dot(centroid_A.T) + centroid_B.T

#     out={}
#     out["R"]=R
#     out["T"]=t
#     out["H"]=createHomog(R, t)
#     return 0
class simulatedRANSAC(slidingWindow):
    def __init__(self,baseWindow=None,cameraSettings=None,frames=2):
        if(baseWindow is not None):
            ####init from previous sliding window
            self.X=copy.deepcopy(baseWindow.X)
            self.kSettings=copy.deepcopy(baseWindow.kSettings)
            self.M=copy.deepcopy(baseWindow.M)
            self.tracks=copy.deepcopy(baseWindow.tracks)
            self.nLandmarks=baseWindow.nLandmarks
            self.nPoses=baseWindow.nPoses
            self.inliers=copy.deepcopy(baseWindow.inliers)
        elif (cameraSettings is not None):
            ####init from scratch
            pass
        self.outliers=[]
    ###########
    ##admin functions
    ###############
    def serializeWindow(self):
        binDiction={}
        binDiction["kSettings"]=pickle.dumps(self.kSettings)
        binDiction["M"]=[]
        for i in self.M:
            binDiction["M"].append(msgpack.packb(i,default=m.encode))
        binDiction["X"]=msgpack.packb(self.X,default=m.encode)
        binDiction["inliers"]=msgpack.dumps(self.inliers)
        binDiction["tracks"]=msgpack.dumps(self.tracks)
        binDiction["nLandmarks"]=self.nLandmarks
        binDiction["nPoses"]=self.nPoses
        binDiction["outliers"]=pickle.dumps(self.outliers)
        return msgpack.dumps(binDiction)
    def deserializeWindow(self,data):
        intern=msgpack.loads(data)
        self.kSettings=pickle.loads(intern["kSettings"])
        self.X=msgpack.unpackb(intern["X"],object_hook=m.decode)
        self.M=[]
        for i in intern["M"]:
            self.M.append(msgpack.unpackb(i,object_hook=m.decode))
        self.inliers=msgpack.loads(intern["inliers"])
        self.tracks=msgpack.loads(intern["tracks"])
        self.nLandmarks=intern["nLandmarks"]
        self.nPoses=intern["nPoses"]
        self.outliers=pickle.loads(intern["outliers"])
    def extractMotion(self,nIterations=150,RMSthreshold=3,resetMotion=True):

        abc=time.time()
        iterations=0
        minimumParams=3
        goodModel=0.8*self.nLandmarks
        bestFit=np.zeros((6,1))
        besterr=np.inf 
        bestInliers=[]
       
        if(resetMotion):
            self.X[0:6,0]=np.zeros(6)

        while iterations < nIterations:
            paramEstimateIndexes,testPointIndexes=self.randomPartition(minimumParams)
            modelEstimationData=self.getSubset(paramEstimateIndexes)
            trainingData=self.getSubset(testPointIndexes)
            


            previousX=np.hstack((modelEstimationData.reprojectLandmark(0)[:,0].reshape(4,1),
                                modelEstimationData.reprojectLandmark(1)[:,0].reshape(4,1),
                                modelEstimationData.reprojectLandmark(2)[:,0].reshape(4,1)))
            currentX=np.hstack((modelEstimationData.reprojectLandmark(0)[:,1].reshape(4,1),
                                modelEstimationData.reprojectLandmark(1)[:,1].reshape(4,1),
                                modelEstimationData.reprojectLandmark(2)[:,1].reshape(4,1)))
            est=rigid_transform_3D(previousX[0:3,:],currentX[0:3,:])
            modelEstimationData.X[0:6,0]=decompose2X(est).reshape(6)
            trainingData.X[0:6,0]=decompose2X(est).reshape(6)

            self.X[0:6,0]=decompose2X(est).reshape(6)
            tempInliers=sorted(list(np.flatnonzero(np.array(trainingData.getAllLandmarkRMS()) <RMSthreshold)))
            currentModelInliers=[]
            for j in tempInliers:
                currentModelInliers.append(testPointIndexes[j])
            newSet=self.getSubset(currentModelInliers)
            if(len(currentModelInliers)>goodModel):
                possibleBetterInliers=currentModelInliers +paramEstimateIndexes
                withInliers=self.getSubset(possibleBetterInliers)

                for i in possibleBetterInliers:

                    previousX=np.hstack((previousX,self.reprojectLandmark(i)[:,0].reshape(4,1)))
                    currentX=np.hstack((currentX,self.reprojectLandmark(i)[:,1].reshape(4,1)))
                possibleBetterEst=rigid_transform_3D(previousX[0:3,:],currentX[0:3,:])
                testData=self.getSubset(possibleBetterInliers)  
                testData.X[0:6,0]=decompose2X(possibleBetterEst).reshape(6)
                betterRMS=testData.getWindowRMS()
                if(betterRMS<besterr):
                    besterr=betterRMS
                    bestInliers=possibleBetterInliers
                    bestFit=decompose2X(possibleBetterEst)
            iterations+=1
        self.X[0:6,0]=copy.deepcopy(bestFit.reshape(6))
        self.inliers=bestInliers
        net=time.time()-abc
        return besterr,net
    def randomPartition(self,minimumParameters=7):
        setOfLandmarks=range(0,self.nLandmarks)
        np.random.shuffle(setOfLandmarks)
        parameterIndexes=sorted(setOfLandmarks[:minimumParameters])
        testPointIdexes=sorted(setOfLandmarks[minimumParameters:])
        return parameterIndexes,testPointIdexes
    #     print(len(range(0,self.getNlandmarks())))
    #     currentXpredictions=self.getReprojections(range(0,self.getNlandmarks()),1)
    #     previousXpredictions=self.getX(range(self.getNlandmarks()))
    #     previousXpredictions=previousXpredictions.reshape((4,len(previousXpredictions)/4),order='F')
    #     while iterations < nIterations:
    #         ##################
    #         ###select a random set of data
    #         #################
    #         paramEstimateIndexes,testPointIndexes=self.randomPartition(minimumParams)
    #         print(paramEstimateIndexes)
    #         est=rigid_transform_3D(previousXpredictions[0:3,paramEstimateIndexes],currentXpredictions[0:3,paramEstimateIndexes])
    #         ########
    #         ##given the new parameter data, estimate the new overall error
    #         #######
    #         testPointsA=previousXpredictions[:,testPointIndexes]
            
    #         testMeasurements=self.M[4:8,testPointIndexes] ##only second frame
    #         testPointsB=est.dot(testPointsA)
            
    #         testPointsB/=testPointsB[3,:]
    #         predictionsBL=self.kSettings["Pl"].dot(testPointsB)
    #         predictionsBL/=predictionsBL[2,:]
    #         predictionsBR=self.kSettings["Pr"].dot(testPointsB)
    #         predictionsBR/=predictionsBR[2,:]
            
            
    #         setPredictions=np.vstack((predictionsBL[0:2,:],predictionsBR[0:2,:]))



    #         diffVect=setPredictions-testMeasurements

    #         L2Norm=np.sqrt(np.square(diffVect).sum(axis=0))
    #         currentModelInliers=sorted(list(np.flatnonzero(L2Norm <RMSthreshold)))
    #         paramEstimateIndexes=np.concatenate((paramEstimateIndexes,currentModelInliers))
    #         if(len(paramEstimateIndexes)>goodModel):
    #             newFit=rigid_transform_3D(previousXpredictions[0:3,paramEstimateIndexes],currentXpredictions[0:3,paramEstimateIndexes])
                
    #             testPointsA=previousXpredictions[:,paramEstimateIndexes]
    #             testMeasurements=self.M[4:8,paramEstimateIndexes] ##only second frame
    #             testPointsB=est.dot(testPointsA)
                
    #             predictionsBL=self.kSettings["Pl"].dot(testPointsB)
    #             predictionsBL/=predictionsBL[2,:]
    #             predictionsBR=self.kSettings["Pr"].dot(testPointsB)
    #             predictionsBR/=predictionsBR[2,:]
            
            
    #             setPredictions=np.vstack((predictionsBL[0:2,:],predictionsBR[0:2,:]))

    #             diffVect=setPredictions-testMeasurements
    #             RMS=np.sqrt((diffVect.ravel()**2).mean())
    #             if besterr > RMS:
    #                 print("new")
    #                 bestFit=newFit
    #                 bestInliers=paramEstimateIndexes
    #                 besterr=RMS
    #         iterations+=1
    #     return bestFit,besterr,bestInliers
    # def randomPartition(self,minimumParameters=7):
    #     setOfLandmarks=range(0,self.getNlandmarks())
    #     np.random.shuffle(setOfLandmarks)
    #     parameterIndexes=setOfLandmarks[:minimumParameters]
    #     testPointIdexes=setOfLandmarks[minimumParameters:]
    #     return parameterIndexes,testPointIdexes

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
        result["R"]=composeR(degrees(ans.x[0]),
                            degrees(ans.x[1]),
                            degrees(ans.x[2]))
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



class simulatedBA(slidingWindow):
    def __init__(self,baseWindow=None,cameraSettings=None,frames=2):
        if(baseWindow is not None):
            ####init from previous sliding window
            self.X=copy.deepcopy(baseWindow.X)
            self.kSettings=copy.deepcopy(baseWindow.kSettings)
            self.M=copy.deepcopy(baseWindow.M)
            self.tracks=copy.deepcopy(baseWindow.tracks)
            self.nLandmarks=baseWindow.nLandmarks
            self.nPoses=baseWindow.nPoses
        elif (cameraSettings is not None):
            ####init from scratch
            pass
        self.outliers=[]
        self.inliers=[]
    def extractMotion(self,resetMotion=True):
        if(resetMotion):
            self.X[0:6,0]=np.zeros(6)
        abc=time.time()
        result=least_squares(self.error,self.X.ravel(),verbose=2,max_nfev=80)
        net=time.time()-abc
        self.X=copy.deepcopy(result.x.reshape(result.x.shape[0],1))
        return self.getWindowRMS(),net
    def error(self,x, *args, **kwargs):
        self.X=copy.deepcopy(x.reshape(6*(self.nPoses-1)+4*self.nLandmarks,1))
        # print(x[0:6],self.getWindowRMS())
        return self.getWindowRMS()
# class slidingWindow(object):
#     def __init__(self,cameraSettings,frames=2):
#         '''
#         SLiding window object of length "frames"
#         X is a  [ Pose1 ]  
#                 [ Pose2 ] 
#                 [ ..... ]6x1 Rtheta vector [Roll pitch yaw X Y Z]^T
#                 [ PoseN ]
#                 [   X1  ]landmark as seen from pose1        [X Y Z W]
#                 [   X2  ]
#                 [   X3  ]
#         shape= [6xNposes + 4xNlandmarks , 1]

#         M is the measurement matrix
#                     [ uleft1    uleft2 ....  uleftn  ]
#           frame1{   [ vleft1    vleft2 ....  vleftn  ]
#                     [ uright1   uright2 .... urightn ]
#                     [ vright1   vright2 .... vrightn ]
#                     [ uleft1    uleft2 ....  uleftn  ]
#           frame2{   [ vleft1    vleft2 ....  vleftn  ]
#                     [ uright1   uright2 .... urightn ]
#                     [ vright1   vright2 .... vrightn ]
#                     [ uleft1    uleft2 ....  uleftn  ]
#           frame3{   [ vleft1    vleft2 ....  vleftn  ]
#                     [ uright1   uright2 .... urightn ]
#                     [ vright1   vright2 .... vrightn ]

#         currently assumes a feature is always tracked i.e no masking available
#         '''
#         self.kSettings=copy.deepcopy(cameraSettings)
#         self.X=np.zeros((0,0),dtype=np.float64) ###[Pose0 Pose1 Pose2|landmarkA landmarkB landmarkC ...]
#         self.M=np.zeros((0,0),dtype=np.float64) ###[Upl0 Vpl0  ]
#         self.tracks=np.zeros((0,0),dtype=np.float64)
#         self.inliers=None
#     def getNlandmarks(self):
#         ######TODO includes the number of frames in this calculation
#         return (self.X.shape[0]-6)/4
#     def getNinliers(self):
#         return np.count(self.inliers)
#     def getNPoseEdge(self,N):
#         poseVector=self.X[N*6:N*6+6]
#         return motionEdge(poseVector[0:3],poseVector[3:6],degrees=False)
#     def getX(self,arrayIndexes):
#         X=np.zeros(len(arrayIndexes)*4)
#         for index in range(0,len(arrayIndexes)):
#             X[4*index:4*index+4]=self.X[6+arrayIndexes[index]*4:6+arrayIndexes[index]*4 + 4]
#         return X
#     def getFeature(self,arrayIndexes,frameNumber):
#         features=np.zeros((4,len(arrayIndexes)))
#         for index in range(0,len(arrayIndexes)):
#             features[0:4,index]=self.M[4*frameNumber:4*frameNumber+4,index]
#         return features
#     def getReprojections(self,arrayIndexes,frameNumber):
#         features=self.getFeature(arrayIndexes,frameNumber)
#         measVect=np.zeros((4,features.shape[1]))
#         measVect[3,:]=np.ones(features.shape[1])
#         measVect[0:2,:]=features[0:2,:]
#         measVect[2,:]=measVect[0,:]-features[2,:]
#         xreproject=self.kSettings["Q"].dot(measVect)
#         xreproject/=xreproject[3,:]
#         return xreproject


# class simulatedRansacWindow(slidingWindow):
#     def __init__(self,Ksettings,initialEdges):
#         super(simulatedRansacWindow,self).__init__(cameraSettings=Ksettings)
#         self.data=copy.deepcopy(initialEdges)
#         self.X=np.zeros((6 + 4*len(self.data.currentEdges)))
#         self.M=np.zeros((4*2,len(self.data.currentEdges)))
#         for edge in range(0,len(self.data.previousEdges)):
#             self.X[edge*4+6:edge*4+4+6]=self.data.previousEdges[edge].X.reshape(4)
#             self.M[0:2,edge]=self.data.previousEdges[edge].L[0:2,0].reshape(2)
#             self.M[2:4,edge]=self.data.previousEdges[edge].R[0:2,0].reshape(2)
#             self.M[4:6,edge]=self.data.currentEdges[edge].L[0:2,0].reshape(2)
#             self.M[6:8,edge]=self.data.currentEdges[edge].R[0:2,0].reshape(2)
#     def RANSACestimate(self,nIterations=5,RMSthreshold=0.0001):
#         iterations=0
#         minimumParams=3
#         goodModel=0.8*self.getNlandmarks()
#         bestFit=None
#         besterr=np.inf 
#         bestInliers=[]

#         print(len(range(0,self.getNlandmarks())))
#         currentXpredictions=self.getReprojections(range(0,self.getNlandmarks()),1)
#         previousXpredictions=self.getX(range(self.getNlandmarks()))
#         previousXpredictions=previousXpredictions.reshape((4,len(previousXpredictions)/4),order='F')
#         while iterations < nIterations:
#             ##################
#             ###select a random set of data
#             #################
#             paramEstimateIndexes,testPointIndexes=self.randomPartition(minimumParams)
#             print(paramEstimateIndexes)
#             est=rigid_transform_3D(previousXpredictions[0:3,paramEstimateIndexes],currentXpredictions[0:3,paramEstimateIndexes])
#             ########
#             ##given the new parameter data, estimate the new overall error
#             #######
#             testPointsA=previousXpredictions[:,testPointIndexes]
            
#             testMeasurements=self.M[4:8,testPointIndexes] ##only second frame
#             testPointsB=est.dot(testPointsA)
            
#             testPointsB/=testPointsB[3,:]
#             predictionsBL=self.kSettings["Pl"].dot(testPointsB)
#             predictionsBL/=predictionsBL[2,:]
#             predictionsBR=self.kSettings["Pr"].dot(testPointsB)
#             predictionsBR/=predictionsBR[2,:]
            
            
#             setPredictions=np.vstack((predictionsBL[0:2,:],predictionsBR[0:2,:]))



#             diffVect=setPredictions-testMeasurements

#             L2Norm=np.sqrt(np.square(diffVect).sum(axis=0))
#             currentModelInliers=sorted(list(np.flatnonzero(L2Norm <RMSthreshold)))
#             paramEstimateIndexes=np.concatenate((paramEstimateIndexes,currentModelInliers))
#             if(len(paramEstimateIndexes)>goodModel):
#                 newFit=rigid_transform_3D(previousXpredictions[0:3,paramEstimateIndexes],currentXpredictions[0:3,paramEstimateIndexes])
                
#                 testPointsA=previousXpredictions[:,paramEstimateIndexes]
#                 testMeasurements=self.M[4:8,paramEstimateIndexes] ##only second frame
#                 testPointsB=est.dot(testPointsA)
                
#                 predictionsBL=self.kSettings["Pl"].dot(testPointsB)
#                 predictionsBL/=predictionsBL[2,:]
#                 predictionsBR=self.kSettings["Pr"].dot(testPointsB)
#                 predictionsBR/=predictionsBR[2,:]
            
            
#                 setPredictions=np.vstack((predictionsBL[0:2,:],predictionsBR[0:2,:]))

#                 diffVect=setPredictions-testMeasurements
#                 RMS=np.sqrt((diffVect.ravel()**2).mean())
#                 if besterr > RMS:
#                     print("new")
#                     bestFit=newFit
#                     bestInliers=paramEstimateIndexes
#                     besterr=RMS
#             iterations+=1
#         return bestFit,besterr,bestInliers
#     def randomPartition(self,minimumParameters=7):
#         setOfLandmarks=range(0,self.getNlandmarks())
#         np.random.shuffle(setOfLandmarks)
#         parameterIndexes=setOfLandmarks[:minimumParameters]
#         testPointIdexes=setOfLandmarks[minimumParameters:]
#         return parameterIndexes,testPointIdexes



# class simulatedWindow(slidingWindow):
#     def __init__(self,Ksettings,initialEdges):
#         super(simulatedWindow,self).__init__(cameraSettings=Ksettings)
#         self.data=copy.deepcopy(initialEdges)
#         self.X=np.zeros((6 + 4*len(self.data.currentEdges)))
#         self.M=np.zeros((4*2,len(self.data.currentEdges)))
#         for edge in range(0,len(self.data.previousEdges)):
#             self.X[edge*4+6:edge*4+4+6]=self.data.previousEdges[edge].X.reshape(4)
#             self.M[0:2,edge]=self.data.previousEdges[edge].L[0:2,0].reshape(2)
#             self.M[2:4,edge]=self.data.previousEdges[edge].R[0:2,0].reshape(2)
#             self.M[4:6,edge]=self.data.currentEdges[edge].L[0:2,0].reshape(2)
#             self.M[6:8,edge]=self.data.currentEdges[edge].R[0:2,0].reshape(2)
#         self.count=0
#     def BAestimate(self):
#         abc=time.time()
#         result=least_squares(self.error,self.X,verbose=2,max_nfev=500)
#         net=time.time()-abc
#         self.X=copy.deepcopy(result.x.reshape(result.x.shape[0],1))
#         return self.getStateRMS(),net
#     def error(self,x, *args, **kwargs):
#         ######
#         ##get camera
#         #####
#         self.count+=1
#         Pal=self.kSettings["Pl"]
#         Par=self.kSettings["Pr"]
#         Pbl=composeCamera(self.kSettings["Pl"][0:3,0:3],copy.deepcopy(x[0:6]))
#         Pbr=composeCamera(self.kSettings["Pl"][0:3,0:3],copy.deepcopy(x[0:6]))
#         errorMatrix=np.zeros(self.M.shape)
#         for i in range(0,self.M.shape[1]):
#             pred1=Pal.dot(x[6+4*i:6+4*i +4].reshape(4,1))
#             pred1/=pred1[2,0]
#             pred2=Par.dot(x[6+4*i:6+4*i +4].reshape(4,1))
#             pred2/=pred2[2,0]
#             pred3=Pbl.dot(x[6+4*i:6+4*i +4].reshape(4,1))
#             pred3/=pred3[2,0]
#             pred4=Pbr.dot(x[6+4*i:6+4*i +4].reshape(4,1))
#             pred4/=pred4[2,0]

#             errorMatrix[0:2,i]=pred1[0:2,0]
#             errorMatrix[2:4,i]=pred2[0:2,0]
#             errorMatrix[4:6,i]=pred3[0:2,0]
#             errorMatrix[6:8,i]=pred4[0:2,0]
#         errorMatrix-=self.M
#         # diff=errorMatrix.reshape(self.M.shape[0]*self.M.shape[1],1)
#         # RMS=np.sqrt((diff**2).mean())
#         return errorMatrix.ravel()
#     def getStateRMS(self):
#         Pal=self.kSettings["Pl"]
#         Par=self.kSettings["Pr"]
#         Pbl=composeCamera(self.kSettings["Pl"][0:3,0:3],self.X[0:6])
#         Pbr=composeCamera(self.kSettings["Pl"][0:3,0:3],self.X[0:6])
#         errorMatrix=np.zeros(self.M.shape)
#         for i in range(0,self.M.shape[1]):
#             pred1=Pal.dot(self.X[6+4*i:6+4*i +4].reshape(4,1))
#             pred1/=pred1[2,0]
#             pred2=Par.dot(self.X[6+4*i:6+4*i +4].reshape(4,1))
#             pred2/=pred2[2,0]
#             pred3=Pbl.dot(self.X[6+4*i:6+4*i +4].reshape(4,1))
#             pred3/=pred3[2,0]
#             pred4=Pbr.dot(self.X[6+4*i:6+4*i +4].reshape(4,1))
#             pred4/=pred4[2,0]

#             errorMatrix[0:2,i]=pred1[0:2,0]
#             errorMatrix[2:4,i]=pred2[0:2,0]
#             errorMatrix[4:6,i]=pred3[0:2,0]
#             errorMatrix[6:8,i]=pred4[0:2,0]
#         errorMatrix-=self.M
#         diff=errorMatrix.reshape(self.M.shape[0]*self.M.shape[1],1)
#         RMS=np.sqrt((diff**2).mean())
#         return RMS      
            
# EPI_THRESHOLD=2.0
# LOWE_THRESHOLD=0.8
# defaultK=np.zeros((3,3),np.float64)
# defaultK[0,0]=803.205
# defaultK[1,1]=803.205
# defaultK[0,2]=433.774
# defaultK[1,2]=468.444
# defaultK[2,2]=1.0

# def singleWindowMatch(currentLandmarks,previousLandmarks):
#     #####perform KNN matching across Frames
#     cvb=CvBridge()
     
#     descTable=descriptorLookUpTable()
#     currentKP=unpackKP(currentLandmarks.leftFeatures)
#     previousKP=unpackKP(previousLandmarks.leftFeatures)
#     currentDescriptors=cvb.imgmsg_to_cv2(currentLandmarks.leftDescr)
#     previousDescriptors=cvb.imgmsg_to_cv2(previousLandmarks.leftDescr)
#     #print(currentDescriptors.shape,previousDescriptors.shape)
#     matcher=getMatcher(descTable[currentLandmarks.descrID]["NormType"])
#     print(type(matcher))
#     ####unpack descriptors
#     ###left Descriptors
#     ans=matcher.knnMatch(currentDescriptors,previousDescriptors,5)
#     epiMatches=loweFilterPotential(ans)
#     # for i in ans:
#     #     indexErrors=[]
#     #     for j in i:
#     #         indexErrors.append(getMatchEpiError(j,currentKP,previousKP))
#     #     bestEpi=min(indexErrors)
#     #     if(bestEpi<EPI_THRESHOLD):
#     #         epiMatches.append(i[indexErrors.index(bestEpi)])
#     # print(float(len(epiMatches))/float(len(currentKP)))

#     ###double check duplicates
#     return epiMatches

# def getNister(currentLandmarks,previousLandmarks,matches,K):
#     currentKP=np.zeros((len(matches),2),dtype=np.float64)
#     previousKP=np.zeros((len(matches),2),dtype=np.float64)
#     print(len(matches),len(currentLandmarks.leftFeatures),len(previousLandmarks.leftFeatures))
#     for i in range(0,len(matches)):
#         currentKP[i,0]=ros2cv_KP(currentLandmarks.leftFeatures[matches[i].queryIdx]).pt[0]
#         currentKP[i,1]=ros2cv_KP(currentLandmarks.leftFeatures[matches[i].queryIdx]).pt[1]
#         previousKP[i,0]=ros2cv_KP(previousLandmarks.leftFeatures[matches[i].trainIdx]).pt[0]
#         previousKP[i,1]=ros2cv_KP(previousLandmarks.leftFeatures[matches[i].trainIdx]).pt[1]
#     print(currentKP.shape,previousKP.shape)
#     E,mask=cv2.findEssentialMat(currentKP,previousKP,K[0,0],(K[0:2,2][0],K[0:2,2][1]),threshold=1)
#     #r1,r2,t=cv2.decomposeEssentialMat(E)
#     print("original",np.count_nonzero(mask))
#     nInliers,R,T,matchMask=cv2.recoverPose(E,currentKP,previousKP,K,mask)
#     ###cheirality check
#     print("Matches MAsk",np.count_nonzero(matchMask))
#     indexes=[]
#     print("here")
#     for i in range(0,len(matchMask)):
#         if(matchMask[i]>0):
#             indexes.append(i)
#     print(indexes)
#     #print(matchMask)
#     print("Nister",nInliers)
#     ###scale
#     ###make homography
#     return createHomog(R,T),matchMask

# def getMatchEpiError(match,leftKP,rightKP):
#     return leftKP[match.queryIdx].pt[1]-rightKP[match.trainIdx].pt[1]


# class BAwindow:
#     def __init__(self,length=2,K=defaultK):
#         self.length=length

# class window:
#     def __init__(self,K,length=2):
#         self.length=length
#         self.window=[]
#         self.motion=[]
#         self.tracks=[]
#         self.motionInliers=[]
#         self.K=K
#         print(K)
#     def update(self,newMsg):
#         if(newMsg.reset):
#             self.window=[]
#             self.motion=[]
#             self.tracks=[]
#             self.motionInliers=[]
#             return windowMatchingResponse()
#         else:
#             self.window.append(newMsg.latestFrame)
#             if(len(self.window)>=self.length+1):
#                 del self.window[0]
#                 del self.motion[0]
#                 del self.tracks[0]
#                 del self.motionInliers[0]
#             if(len(self.window)>=self.length):
#                 self.tracks.append(singleWindowMatch(self.window[-1],self.window[-2]))
#                 h,m=getNister(self.window[-1],self.window[-2],self.tracks[-1],self.K)
#                 self.motion.append(h)
#                 self.motionInliers.append(m)
#                 print(len(self.window),len(self.motion),len(self.tracks))
#             return self.getStatus()
#     def getStatus(self):
#         output=windowMatchingResponse()
#         cvb=CvBridge()
#         for i in self.window:
#             output.state.msgs.append(i)
#         for i in self.motion:
#             output.state.motion.append(cvb.cv2_to_imgmsg(i))
#         for i in self.tracks:
#             msg=interFrameTracks()
#             for j in i:
#                 msg.tracks.append(cv2ros_dmatch(j))
#             output.state.tracks.append(msg)
#         for i in range(0,len(self.motionInliers)):
#             outMask=np.zeros((1,len(self.motionInliers[i])),dtype=np.int8)
#             for j in range(0,len(self.motionInliers[i])):
#                 if(self.motionInliers[i][j]>0):
#                     outMask[0,j]=1
#             output.state.tracks[i].motionInliers=cvb.cv2_to_imgmsg(outMask)
#         return output
    

# def triangulate(leftKP,rightKP,Q):
#     for i in range(0,len(leftKP)):
#         dVector=np.zeros((4,1),dtype=np.float64)
#         dVector[0,0]=leftKP[i].pt[0]
#         dVector[1,0]=leftKP[i].pt[1]
#         dVector[2,0]=leftKP[i].pt[0]-rightKP[i].pt[0]
#         dVector[3,0]=1.0

# ###################
# ###Algorithm One

# def getEpiPolarMatches(leftKP,rightKP):
#     ##build Distance Table
#     mask=np.zeros((len(leftKP),len(rightKP)),dtype=np.uint8)
#     distances=np.zeros((len(leftKP),len(rightKP)),dtype=np.float64)
#     for row in range(0,distances.shape[0]):
#         for col in range(0,distances.shape[1]):
#             distances[row,col]=abs(leftKP[row].pt[1]-rightKP[col].pt[1])
#     for row in range(0,distances.shape[0]):
#         for col in range(0,distances.shape[1]):
#             if(distances[row,col]<=EPI_THRESHOLD):
#                 mask[row,col]=1
#     return mask,distances

# def loweFilterPotential(matches):
#     goodMatches=[]
#     for i in matches:
#         if(len(i)==1):
#             goodMatches.append(i[0])
#         elif(len(i)>1):
#             if(i[0].distance<LOWE_THRESHOLD*i[1].distance):
#                 goodMatches.append(i[0])
#     return goodMatches

# def getPotentialMatches(leftDescr,rightDescr,mask,norm):
#     matcher=getMatcher(norm)
#     ####unpack descriptors
#     ###left Descriptors
#     ans=matcher.knnMatch(leftDescr,rightDescr,2,mask)
#     return ans
# def algorithm_one(stereoFrame):
#     cvb=CvBridge()
#     descTable=descriptorLookUpTable()

#     print("received ",stereoFrame.detID,stereoFrame.descrID)
#     ###unpack the keypoints into cv lists
#     lkp=unpackKP(stereoFrame.leftFeatures)
#     assignIDs(lkp)
#     rkp=unpackKP(stereoFrame.rightFeatures)
#     assignIDs(rkp)
#     ld=cvb.imgmsg_to_cv2(stereoFrame.leftDescr)
#     rd=cvb.imgmsg_to_cv2(stereoFrame.rightDescr)
#     ###filter by epipolar matches

#     print("Matching")
#     epiTime=ProcTime()
#     epiTime.label="Epipolar Filter"
#     MatchTime=ProcTime()
#     MatchTime.label="KNN Match"
#     loweTime=ProcTime()
#     loweTime.label="lowe Ratio"

#     startTime=time.time()
#     mask,dist=getEpiPolarMatches(lkp,rkp)
#     epiTime.seconds=time.time()-startTime


#     startTime=time.time()
#     initialMatches=getPotentialMatches(ld,
#                           rd,
#                           mask,descTable[stereoFrame.descrID]["NormType"])
#     MatchTime.seconds=time.time()-startTime

#     startTime=time.time()
#     finalMatches=loweFilterPotential(initialMatches)
#     loweTime.seconds=time.time()-startTime

#     reply=stereoMatchingResponse()
#     #####pack into a frame
#     msg=stereoLandmarks()
#     msg.detID=stereoFrame.detID
#     msg.descrID=stereoFrame.descrID
#     newLdesc=np.zeros((len(finalMatches),ld.shape[1]),dtype=ld.dtype)
#     newRdesc=np.zeros((len(finalMatches),ld.shape[1]),dtype=ld.dtype)
#     for index in range(0,len(finalMatches)):
#         ###pack left 
#         msg.leftFeatures.append(cv2ros_KP(lkp[finalMatches[index].queryIdx]))
#         newLdesc[index,:]=ld[finalMatches[index].queryIdx,:]
#         ###pack right
#         msg.rightFeatures.append(cv2ros_KP(rkp[finalMatches[index].trainIdx]))
#         newRdesc[index,:]=rd[finalMatches[index].trainIdx,:]
#         ###pack match
#         match=cv2.DMatch()
#         match.distance=finalMatches[index].distance
#         match.queryIdx=index
#         match.trainIdx=index
#         match.imgIdx=0
#         msg.matches.append(cv2ros_dmatch(match))
#     msg.leftDescr=cvb.cv2_to_imgmsg(newLdesc)
#     msg.rightDescr=cvb.cv2_to_imgmsg(newRdesc)
#     print(len(msg.leftFeatures),len(msg.rightFeatures))
#     reply.out=msg
#     reply.out.proc.append(epiTime)
#     reply.out.proc.append(MatchTime)
#     reply.out.proc.append(loweTime)
#     return reply

###################
###Algorithm Two

########
###Window matching

#######
##Algorithm One

#################
###Motion Extraction

class BAextractor:
    def __init__(self,nIter=500):
        self.bumblebee=stereoCamera()
        self.maxIterations=nIter
    def extract(self,currentPoints,currentLandmarks,
                previousPoints,previousLandmarks):
        ans=least_squares(self.error,np.array([0,0,0,0,0,0]),verbose=False,max_nfev=self.maxIterations,
                            args=(currentPoints,currentLandmarks,
                        previousPoints,previousLandmarks))
        result={}
        angles=ans.x[0:3].reshape(1,3)
        angles[0,0]=angles[0,0]
        angles[0,1]=angles[0,1]
        angles[0,2]=angles[0,2]
        T=ans.x[3:].reshape(3,1)
        H=createHomog(composeR(degrees(angles[0,0]),degrees(angles[0,1]),degrees(angles[0,2])),T)
        return H,ans
    def error(self,x, *args, **kwargs):
        Rest=composeR(x[0],x[1],x[2],False)
        T=x[3:]
        Ht=createHomog(Rest,T)
        totalRMS=0
        index=0
        P=np.zeros((3,4),dtype=np.float64)
        P[0:3,0:3]=Rest
        P[0:3,3]=T 
        Pb= self.bumblebee.kSettings["k"].dot(P)
        for dataIndex in range(0,len(args[0])):
            newPixel=Pb.dot(args[1][dataIndex])
            newPixel= newPixel/newPixel[2,0]
            e=abs(args[2][dataIndex][0]-newPixel[0,0])+abs(args[2][dataIndex][1]-newPixel[1,0])
            totalRMS+=e
            index+=1
        return totalRMS

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
        # dot is matrix multiplication for array
        H = np.transpose(AA).dot(BB)
        U, S, Vt = np.linalg.svd(H)

        R = Vt.T * U.T

        # special reflection case
        if(np.linalg.det(R) < 0):
            Vt[2,:] *= -1
            R = Vt.T * U.T
        t = -R.dot(centroid_A.T) + centroid_B.T
        out={}
        out["R"]=R
        out["T"]=t
        out["H"]=createHomog(R, t)
        return out
    def RANSAC_rigid_body(self,currentPoints,previousPoints,maxIterations=100):
        currentIteration=0
        out={}
        currentSetInliers=[]
        while(currentIteration<maxIterations):
            randomSubset=random.sample(range(0,len(currentPoints)),8)
            possibleCurrentPoints=[currentPoints[i] for i in randomSubset]
            possiblePreviousPoints=[previousPoints[i] for i in randomSubset]
            hEst=self.rigid_transform_3D(possiblePreviousPoints,possibleCurrentPoints)
            currentIteration+=1
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






class stereoWindow:
    def __init__(self):
        getCameraSettingsFromServer(full=False)