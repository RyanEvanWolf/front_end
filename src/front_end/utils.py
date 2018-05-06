import numpy as np
import cv2
from statistics import mean,stdev
from front_end.msg import kPoint,cvMatch


def getFAST_parameters():
    threshold=np.arange(1, 60, 3)
    dType=(cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    maxSuppression=(True,False)
    output={}
    output["Threshold"]=threshold
    output["dType"]=dType
    output["NonMaximumSuppression"]=maxSuppression
    return output

def getFAST_Combinations():
    output={}
    output["Name"]="FAST"
    params=getFAST_parameters()
    orderedKeys=sorted(params.keys())
    print(orderedKeys)
    return output


# def getFAST_Attributes():
#     threshold=np.arange(1, 60, 3)
#     dType=(cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
#     maxSuppression=(True,False)
#     ###pack into string format
#     output={}
#     output["Threshold"]=threshold
#     output["dType"]=dType
#     output["NonMaximumSuppression"]=maxSuppression
#     packedStrings=[]
#     for t in threshold:
#         for d in dType:
#             for m in maxSuppression:
#                 msg="Threshold,"+str(t)+",dType,"+str(d)+",NonMaximumSuppression,"+str(m)
#                 packedStrings.append(msg)
#     return output,packedStrings

# def getBRIEF_Attributes():
#     size=[16,32,64]
#     orientation=[1,0]
#     output={}
#     output["bytes"]=size
#     output["use_orientation"]=orientation
#     packedStrings=[]
#     for s in size:
#         for o in orientation:
#             msg="BRIEF,bytes,"+str(s)+",use_orientation,"+str(o)
#             packedStrings.append(msg)
#     return output,packedStrings

# def getAllDescriptor_Attributes():
#     descriptorStrings=[]
#     ########
#     fake,strings=getBRIEF_Attributes()
#     descriptorStrings+=strings

#     fake,fake2,strings=getSURF_Attributes()
#     descriptorStrings+=strings
#     return descriptorStrings

# def getSURF_Attributes():
#     threshold=np.arange(200,500,50)
#     nOctave=np.arange(4,7,1)
#     nOctaveLayers=np.arange(3,6,1)
#     extended=(1,0)
#     upright=(1,0)
#     ###pack into string format
#     output={}
#     output["HessianThreshold"]=threshold
#     output["nOctave"]=nOctave
#     output["nOctaveLayers"]=nOctaveLayers
#     output["Extended"]=extended
#     output["Upright"]=upright
#     detectorStrings=[]
#     descriptorStrings=[]
#     for t in threshold:
#         for n in nOctave:
#             for nl in nOctaveLayers:
#                 for e in extended:
#                     for u in upright:
#                         msg="HessianThreshold,"+str(t)+",nOctave,"+str(n)+",nOctaveLayers,"+str(nl)+",Extended,"+str(e)+",Upright,"+str(u)
#                         detectorStrings.append(msg)
#     ##descriptor Settings
#     for e in extended:
#         for u in upright:
#             msg="SURF,HessianThreshold,"+str(threshold[0])+",nOctave,"+str(nOctave[0])+",nOctaveLayers,"+str(nOctaveLayers[1])+",Extended,"+str(e)+",Upright,"+str(u)
#             descriptorStrings.append(msg)
#     return output,detectorStrings,descriptorStrings

# def updateDetector(name,csvString,detectorRef):
#     parts=csvString.split(",")
#     if(name=="FAST"):
#         detectorRef.setThreshold(int(parts[1]))
#         detectorRef.setType(int(parts[3]))
#         detectorRef.setNonmaxSuppression(bool(parts[5]))
#     if(name=="SURF"):
#         detectorRef.setHessianThreshold(float(parts[1]))
#         detectorRef.setNOctaves(int(parts[3]))
#         detectorRef.setNOctaveLayers(int(parts[5]))
#         detectorRef.setExtended(int(parts[7]))
#         detectorRef.setUpright(int(parts[9]))

# def updateDescriptor(csvString,detectorRef):
#     parts=csvString.split(",")

# def getDetector(name):
#     if(name=="FAST"):
#         return True,cv2.FastFeatureDetector_create()
#     elif(name=="SURF"):
#         return True,cv2.xfeatures2d.SURF_create()
#     else:
#         return False,None

# def getDescriptor(csvString):
#     parts=csvString.split(",")
#     name=parts[0]
#     if(name=="BRIEF"):
#         bits=int(parts[2])
#         orientation=int(parts[4])
#         return True,cv2.xfeatures2d.BriefDescriptorExtractor_create(bits,orientation)
#     elif(name=="SURF"):
#         threshold=float(parts[2])
#         octaves=int(parts[4])
#         layers=int(parts[6])
#         extended=int(parts[8])
#         upright=int(parts[10])

#         return True,cv2.xfeatures2d.SURF_create(threshold,octaves,
#                                                 layers,extended,upright)
#     else:
#         return False,None
def getKPstats(KPset):
    ##get X list
    x=[]
    y=[]
    for i in KPset:
        x.append(i.pt[0])
        y.append(i.pt[1])
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

def packKP(KPlist):
    newList=[]
    for i in KPlist:
        newList.append(cv2ros_KP(i))
    return newList

def unpackKP(messagelist):
    newList=[]
    for i in messagelist:
        newList.append(ros2cv_KP(i))
    return newList

def ros2cv_KP(message):
    kp=cv2.KeyPoint()
    kp.angle=message.angle
    kp.octave=message.octave
    kp.pt=(message.x,message.y)
    kp.response=message.response
    kp.size=message.size
    kp.class_id=message.class_id
    return kp

def cv2ros_KP(kp):
    message=kPoint()
    message.angle=kp.angle
    message.octave=kp.octave
    message.x=kp.pt[0]
    message.y=kp.pt[1]
    message.response=kp.response
    message.size=kp.size
    message.class_id=kp.class_id
    return message

def cv2ros_dmatch(dm):
    msg=cvMatch()
    msg.imgIdx=dm.imgIdx
    msg.trainIdx=dm.trainIdx
    msg.queryIdx=dm.queryIdx
    msg.distance=dm.distance
    return msg

def ros2cv_dmatch(msg):
    dm=cv2.DMatch()
    dm.imgIdx=msg.imgIdx
    dm.trainIdx=msg.trainIdx
    dm.queryIdx=msg.queryIdx
    dm.distance=msg.distance
    return dm 

def printFormattedKP(kp):
    out=""
    out+="pt,"+str(kp.pt)
    out+=",angle,"+str(kp.angle)
    out+=",octave,"+str(kp.octave)
    out+=",response,"+str(kp.response)
    out+=",size,"+str(kp.size)
    out+=",class_id,"+str(kp.class_id)
    return out


# def JSON_to_cvKP(JSON):
#     pass

# def getHomogZeros():
#     out=np.zeros((4,1),dtype=np.float64)
#     out[3,0]=1
#     return out

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


