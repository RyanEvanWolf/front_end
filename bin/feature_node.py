#!/usr/bin/env python
#####ros related imports
import rospy 
import cv2
import time

from front_end.utils import *
from front_end.srv import *
from front_end.msg import frameExtract,ProcTime,kPoint

from cv_bridge import CvBridge
import inspect
cvb=CvBridge()

def singleImageExtraction_fn(req):
    leftImg=cvb.imgmsg_to_cv2(req.leftImg)
    rightImg=cvb.imgmsg_to_cv2(req.rightImg)
    ###update Detector
    success,detector=getDetector(req.detectorName)
    success,descriptor=getDetector(req.descriptorName)
    updateDetector(req.descriptorName,req.desc_attrib[0],descriptor)
    processCount=0

    ltime=ProcTime()
    ltime.label="lKP"

    ldtime=ProcTime()
    ldtime.label="lD"

    rtime=ProcTime()
    rtime.label="rKP"

    rdtime=ProcTime()
    rdtime.label="rD"

    fullOutputMessage=singleImageExtractionResponse()
    for settingsIndex in req.det_attrib:
        updateDetector(req.detectorName,settingsIndex,detector)

        latestFrame=frameExtract()
        startTime=time.time()
        leftKPTs=detector.detect(leftImg)
        ltime.seconds=time.time()-startTime

        startTime=time.time()
        rightKPTs=detector.detect(rightImg)
        rtime.seconds=time.time()-startTime
        ###pack into frames
        for l in leftKPTs:
            kp=kPoint()
            kp.angle=l.angle
            kp.octave=l.octave
            kp.x=l.pt[1]
            kp.y=l.pt[0]
            kp.response=l.response
            kp.size=l.size
            kp.class_id=l.class_id
            latestFrame.leftFeatures.append(kp)
        for r in rightKPTs:
            kp=kPoint()
            kp.angle=r.angle
            kp.octave=r.octave
            kp.x=r.pt[1]
            kp.y=r.pt[0]
            kp.response=r.response
            kp.size=r.size
            kp.class_id=r.class_id
            latestFrame.rightFeatures.append(kp)
        for descriptorIndex in req.desc_attrib:
            updateDetector(req.descriptorName,descriptorIndex,descriptor)
            ldtime.seconds=0
            if(len(leftKPTs)>0):
                startTime=time.time()
                fake,leftDescr=descriptor.compute(leftImg,leftKPTs)
                ldtime.seconds=time.time()-startTime
                ##pack into frame
                latestFrame.ldescr.append(cvb.cv2_to_imgmsg(leftDescr))
            if(len(rightKPTs)>0):
                startTime=time.time()
                fake,rightDescr=descriptor.compute(rightImg,rightKPTs)
                rdtime.seconds=time.time()-startTime
                ##pack into frame
                latestFrame.rdescr.append(cvb.cv2_to_imgmsg(rightDescr))
            processCount+=1
            print(str(processCount)+"/"+str(len(req.det_attrib)*len(req.desc_attrib)))
        fullOutputMessage.outputFrames.append(latestFrame)

        
    print("-----")
    return fullOutputMessage

if __name__ == '__main__':
    rospy.init_node("feature_node")
    setServices=[]
    s=rospy.Service("feature_node/singleImageExtraction",singleImageExtraction,singleImageExtraction_fn)

    rospy.spin()