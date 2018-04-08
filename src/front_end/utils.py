import numpy as np
import cv2
from statistics import mean,stdev



def getFAST_Attributes():
    threshold=np.arange(1, 60, 3)
    dType=(cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    maxSuppression=(True,False)
    ###pack into string format
    output={}
    output["Threshold"]=threshold
    output["dType"]=dType
    output["NonMaximumSuppression"]=maxSuppression
    packedStrings=[]
    for t in threshold:
        for d in dType:
            for m in maxSuppression:
                msg="Threshold,"+str(t)+",dType,"+str(d)+",NonMaximumSuppression,"+str(m)
                packedStrings.append(msg)
    return output,packedStrings

def getSURF_Attributes():
    threshold=np.arange(200,500,50)
    nOctave=np.arange(4,7,1)
    nOctaveLayers=np.arange(3,6,1)
    extended=(1,0)
    upright=(1,0)
    ###pack into string format
    output={}
    output["HessianThreshold"]=threshold
    output["nOctave"]=nOctave
    output["nOctaveLayers"]=nOctaveLayers
    output["Extended"]=extended
    output["Upright"]=upright
    detectorStrings=[]
    descriptorStrings=[]
    for t in threshold:
        for n in nOctave:
            for nl in nOctaveLayers:
                for e in extended:
                    for u in upright:
                        msg="HessianThreshold,"+str(t)+",nOctave,"+str(n)+",nOctaveLayers,"+str(nl)+",Extended,"+str(e)+",Upright,"+str(u)
                        detectorStrings.append(msg)
    ##descriptor Settings
    for e in extended:
        for u in upright:
            msg="HessianThreshold,"+str(threshold[0])+",nOctave,"+str(nOctave[0])+",nOctaveLayers,"+str(nOctaveLayers[1])+",Extended,"+str(e)+",Upright,"+str(u)
            descriptorStrings.append(msg)
    return output,detectorStrings,descriptorStrings

def updateDetector(name,csvString,detectorRef):
    parts=csvString.split(",")
    if(name=="FAST"):
        detectorRef.setThreshold(int(parts[1]))
        detectorRef.setType(int(parts[3]))
        detectorRef.setNonmaxSuppression(bool(parts[5]))
    if(name=="SURF"):
        detectorRef.setHessianThreshold(float(parts[1]))
        detectorRef.setNOctaves(int(parts[3]))
        detectorRef.setNOctaveLayers(int(parts[5]))
        detectorRef.setExtended(int(parts[7]))
        detectorRef.setUpright(int(parts[9]))

def getDetector(name):
    if(name=="FAST"):
        return True,cv2.FastFeatureDetector_create()
    elif(name=="SURF"):
        return True,cv2.xfeatures2d.SURF_create()
    else:
        return False,None

def getKPstats(KPset):
    ##get X list
    x=[]
    y=[]
    for i in KPset:
        x.append(i.pt[1])
        y.append(i.pt[0])
    if(len(x)>2):
        xavg=mean(x)
        xdev=stdev(x)
    else:
        xavg=0
        xdev=0
    if(len(y)>2):
        yavg=mean(y)
        ydev=stdev(y)
    else:
        yavg=0
        ydev=0
    return {"X":{"Avg":xavg,"stdDev":xdev},"Y":{"Avg":yavg,"stdDev":ydev}}
def cvKP_to_JSON(kp):
    pass 

def JSON_to_cvKP(JSON):
    pass

# def getOrbParameters():
#     ORB_Messages = []
#     scaleVect =np.linspace(1.0, 2.5, 4, endpoint=True)#,8
#     edgeVect = np.arange(32, 64, 16)#16,64,16
#     levelVect = np.arange(2, 10, 4)#2,10,2
#     wtaVect = np.arange(2, 4, 1)
#     scoreVect = [cv2.ORB_HARRIS_SCORE, cv2.ORB_FAST_SCORE]
#     patchVect =edgeVect
#     for sc in scaleVect:
#         for pat in patchVect:
#             for ed in edgeVect:
#                 for wt in wtaVect:
#                     for l in levelVect:
#                         for scor in scoreVect:
#                             newSettings = ORB()
#                             newSettings.maxFeatures.data=10000
#                             newSettings.scale.data = sc
#                             newSettings.edge.data = ed
#                             newSettings.level.data = l
#                             newSettings.wta.data = wt
#                             newSettings.score.data = scor
#                             newSettings.patch.data = pat
#                             ORB_Messages.append(newSettings)
#     return ORB_Messages


