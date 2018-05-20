import numpy as np
from front_end.utils import *
from front_end.features import *
from front_end.srv import *
from cv_bridge import CvBridge
from front_end.msg import ProcTime,kPoint,stereoLandmarks,interFrameTracks
from front_end.motion import createHomog


EPI_THRESHOLD=2.0
LOWE_THRESHOLD=0.8
defaultK=np.zeros((3,3),np.float64)
defaultK[0,0]=803.205
defaultK[1,1]=803.205
defaultK[0,2]=433.774
defaultK[1,2]=468.444
defaultK[2,2]=1.0

def singleWindowMatch(currentLandmarks,previousLandmarks):
    #####perform KNN matching across Frames
    cvb=CvBridge()
     
    descTable=descriptorLookUpTable()
    currentKP=unpackKP(currentLandmarks.leftFeatures)
    previousKP=unpackKP(previousLandmarks.leftFeatures)
    currentDescriptors=cvb.imgmsg_to_cv2(currentLandmarks.leftDescr)
    previousDescriptors=cvb.imgmsg_to_cv2(previousLandmarks.leftDescr)
    #print(currentDescriptors.shape,previousDescriptors.shape)
    matcher=getMatcher(descTable[currentLandmarks.descrID]["NormType"])
    print(type(matcher))
    ####unpack descriptors
    ###left Descriptors
    ans=matcher.knnMatch(currentDescriptors,previousDescriptors,5)
    epiMatches=loweFilterPotential(ans)
    # for i in ans:
    #     indexErrors=[]
    #     for j in i:
    #         indexErrors.append(getMatchEpiError(j,currentKP,previousKP))
    #     bestEpi=min(indexErrors)
    #     if(bestEpi<EPI_THRESHOLD):
    #         epiMatches.append(i[indexErrors.index(bestEpi)])
    # print(float(len(epiMatches))/float(len(currentKP)))

    ###double check duplicates
    return epiMatches

def getNister(currentLandmarks,previousLandmarks,matches,K):
    currentKP=np.zeros((len(matches),2),dtype=np.float64)
    previousKP=np.zeros((len(matches),2),dtype=np.float64)
    print(len(matches),len(currentLandmarks.leftFeatures),len(previousLandmarks.leftFeatures))
    for i in range(0,len(matches)):
        currentKP[i,0]=ros2cv_KP(currentLandmarks.leftFeatures[matches[i].queryIdx]).pt[0]
        currentKP[i,1]=ros2cv_KP(currentLandmarks.leftFeatures[matches[i].queryIdx]).pt[1]
        previousKP[i,0]=ros2cv_KP(previousLandmarks.leftFeatures[matches[i].trainIdx]).pt[0]
        previousKP[i,1]=ros2cv_KP(previousLandmarks.leftFeatures[matches[i].trainIdx]).pt[1]
    print(currentKP.shape,previousKP.shape)
    E,mask=cv2.findEssentialMat(currentKP,previousKP,K[0,0],(K[0:2,2][0],K[0:2,2][1]),threshold=1)
    #r1,r2,t=cv2.decomposeEssentialMat(E)
    print("original",np.count_nonzero(mask))
    nInliers,R,T,matchMask=cv2.recoverPose(E,currentKP,previousKP,K,mask)
    ###cheirality check
    print("Matches MAsk",np.count_nonzero(matchMask))
    indexes=[]
    print("here")
    for i in range(0,len(matchMask)):
        if(matchMask[i]>0):
            indexes.append(i)
    print(indexes)
    #print(matchMask)
    print("Nister",nInliers)
    ###scale
    ###make homography
    return createHomog(R,T),matchMask

def getMatchEpiError(match,leftKP,rightKP):
    return leftKP[match.queryIdx].pt[1]-rightKP[match.trainIdx].pt[1]


class BAwindow:
    def __init__(self,length=2,K=defaultK):
        self.length=length

class window:
    def __init__(self,K,length=2):
        self.length=length
        self.window=[]
        self.motion=[]
        self.tracks=[]
        self.motionInliers=[]
        self.K=K
        print(K)
    def update(self,newMsg):
        if(newMsg.reset):
            self.window=[]
            self.motion=[]
            self.tracks=[]
            self.motionInliers=[]
            return windowMatchingResponse()
        else:
            self.window.append(newMsg.latestFrame)
            if(len(self.window)>=self.length+1):
                del self.window[0]
                del self.motion[0]
                del self.tracks[0]
                del self.motionInliers[0]
            if(len(self.window)>=self.length):
                self.tracks.append(singleWindowMatch(self.window[-1],self.window[-2]))
                h,m=getNister(self.window[-1],self.window[-2],self.tracks[-1],self.K)
                self.motion.append(h)
                self.motionInliers.append(m)
                print(len(self.window),len(self.motion),len(self.tracks))
            return self.getStatus()
    def getStatus(self):
        output=windowMatchingResponse()
        cvb=CvBridge()
        for i in self.window:
            output.state.msgs.append(i)
        for i in self.motion:
            output.state.motion.append(cvb.cv2_to_imgmsg(i))
        for i in self.tracks:
            msg=interFrameTracks()
            for j in i:
                msg.tracks.append(cv2ros_dmatch(j))
            output.state.tracks.append(msg)
        for i in range(0,len(self.motionInliers)):
            outMask=np.zeros((1,len(self.motionInliers[i])),dtype=np.int8)
            for j in range(0,len(self.motionInliers[i])):
                if(self.motionInliers[i][j]>0):
                    outMask[0,j]=1
            output.state.tracks[i].motionInliers=cvb.cv2_to_imgmsg(outMask)
        return output
    

def triangulate(leftKP,rightKP,Q):
    for i in range(0,len(leftKP)):
        dVector=np.zeros((4,1),dtype=np.float64)
        dVector[0,0]=leftKP[i].pt[0]
        dVector[1,0]=leftKP[i].pt[1]
        dVector[2,0]=leftKP[i].pt[0]-rightKP[i].pt[0]
        dVector[3,0]=1.0

###################
###Algorithm One

def getEpiPolarMatches(leftKP,rightKP):
    ##build Distance Table
    mask=np.zeros((len(leftKP),len(rightKP)),dtype=np.uint8)
    distances=np.zeros((len(leftKP),len(rightKP)),dtype=np.float64)
    for row in range(0,distances.shape[0]):
        for col in range(0,distances.shape[1]):
            distances[row,col]=abs(leftKP[row].pt[1]-rightKP[col].pt[1])
    for row in range(0,distances.shape[0]):
        for col in range(0,distances.shape[1]):
            if(distances[row,col]<=EPI_THRESHOLD):
                mask[row,col]=1
    return mask,distances

def loweFilterPotential(matches):
    goodMatches=[]
    for i in matches:
        if(len(i)==1):
            goodMatches.append(i[0])
        elif(len(i)>1):
            if(i[0].distance<LOWE_THRESHOLD*i[1].distance):
                goodMatches.append(i[0])
    return goodMatches

def getPotentialMatches(leftDescr,rightDescr,mask,norm):
    matcher=getMatcher(norm)
    ####unpack descriptors
    ###left Descriptors
    ans=matcher.knnMatch(leftDescr,rightDescr,2,mask)
    return ans
def algorithm_one(stereoFrame):
    cvb=CvBridge()
    descTable=descriptorLookUpTable()

    print("received ",stereoFrame.detID,stereoFrame.descrID)
    ###unpack the keypoints into cv lists
    lkp=unpackKP(stereoFrame.leftFeatures)
    assignIDs(lkp)
    rkp=unpackKP(stereoFrame.rightFeatures)
    assignIDs(rkp)
    ld=cvb.imgmsg_to_cv2(stereoFrame.leftDescr)
    rd=cvb.imgmsg_to_cv2(stereoFrame.rightDescr)
    ###filter by epipolar matches

    print("Matching")
    epiTime=ProcTime()
    epiTime.label="Epipolar Filter"
    MatchTime=ProcTime()
    MatchTime.label="KNN Match"
    loweTime=ProcTime()
    loweTime.label="lowe Ratio"

    startTime=time.time()
    mask,dist=getEpiPolarMatches(lkp,rkp)
    epiTime.seconds=time.time()-startTime


    startTime=time.time()
    initialMatches=getPotentialMatches(ld,
                          rd,
                          mask,descTable[stereoFrame.descrID]["NormType"])
    MatchTime.seconds=time.time()-startTime

    startTime=time.time()
    finalMatches=loweFilterPotential(initialMatches)
    loweTime.seconds=time.time()-startTime

    reply=stereoMatchingResponse()
    #####pack into a frame
    msg=stereoLandmarks()
    msg.detID=stereoFrame.detID
    msg.descrID=stereoFrame.descrID
    newLdesc=np.zeros((len(finalMatches),ld.shape[1]),dtype=ld.dtype)
    newRdesc=np.zeros((len(finalMatches),ld.shape[1]),dtype=ld.dtype)
    for index in range(0,len(finalMatches)):
        ###pack left 
        msg.leftFeatures.append(cv2ros_KP(lkp[finalMatches[index].queryIdx]))
        newLdesc[index,:]=ld[finalMatches[index].queryIdx,:]
        ###pack right
        msg.rightFeatures.append(cv2ros_KP(rkp[finalMatches[index].trainIdx]))
        newRdesc[index,:]=rd[finalMatches[index].trainIdx,:]
        ###pack match
        match=cv2.DMatch()
        match.distance=finalMatches[index].distance
        match.queryIdx=index
        match.trainIdx=index
        match.imgIdx=0
        msg.matches.append(cv2ros_dmatch(match))
    msg.leftDescr=cvb.cv2_to_imgmsg(newLdesc)
    msg.rightDescr=cvb.cv2_to_imgmsg(newRdesc)
    print(len(msg.leftFeatures),len(msg.rightFeatures))
    reply.out=msg
    reply.out.proc.append(epiTime)
    reply.out.proc.append(MatchTime)
    reply.out.proc.append(loweTime)
    return reply

###################
###Algorithm Two

########
###Window matching

#######
##Algorithm One
