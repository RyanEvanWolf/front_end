#!/usr/bin/env python
#####ros related imports
import rospy 
import cv2
import time

from front_end.utils import *
from front_end.srv import *
from front_end.msg import frameDetection,ProcTime,kPoint

from cv_bridge import CvBridge

cvb=CvBridge()

global imagesProcessed
imagesProcessed=0

def singleImageDetection_fn(req):
    global imagesProcessed
    leftImg=cvb.imgmsg_to_cv2(req.leftImg)
    rightImg=cvb.imgmsg_to_cv2(req.rightImg)
    ###update Detector

    success,detector=getDetector(req.detectorName)
    #success,descriptor=getDetector(req.descriptorName)
    processCount=0

    ltime=ProcTime()
    ltime.label="lKP"

    rtime=ProcTime()
    rtime.label="rKP"

    fullOutputMessage=singleImageDetectionResponse()
    for settingsIndex in req.det_attrib:
        updateDetector(req.detectorName,settingsIndex,detector)

        latestFrame=frameDetection()
        startTime=time.time()
        leftKPTs=detector.detect(leftImg)
        ltime.seconds=time.time()-startTime

        startTime=time.time()
        rightKPTs=detector.detect(rightImg)
        rtime.seconds=time.time()-startTime
        ###pack into frames
        lstat=getKPstats(leftKPTs)
        rstat=getKPstats(rightKPTs)
        latestFrame.l_xAvg=lstat["X"]["Avg"]
        latestFrame.l_yAvg=lstat["Y"]["Avg"]
        latestFrame.l_xStd=lstat["X"]["stdDev"]
        latestFrame.l_yStd=lstat["Y"]["stdDev"]
        latestFrame.nLeft=len(leftKPTs)

        latestFrame.r_xAvg=rstat["X"]["Avg"]
        latestFrame.r_yAvg=rstat["Y"]["Avg"]
        latestFrame.r_xStd=rstat["X"]["stdDev"]
        latestFrame.r_yStd=rstat["Y"]["stdDev"]
        latestFrame.nRight=len(rightKPTs)
        if(req.returnKP):
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
        latestFrame.processingTime.append(ltime)
        latestFrame.processingTime.append(rtime)
        processCount+=1
        # for descriptorIndex in req.desc_attrib:
        #     updateDetector(req.descriptorName,descriptorIndex,descriptor)
        #     ldtime.seconds=0
        #     if(len(leftKPTs)>0):
        #         startTime=time.time()
        #         fake,leftDescr=descriptor.compute(leftImg,leftKPTs)
        #         ldtime.seconds=time.time()-startTime
        #         ##pack into frame
        #         latestFrame.ldescr.append(cvb.cv2_to_imgmsg(leftDescr))
        #     if(len(rightKPTs)>0):
        #         startTime=time.time()
        #         fake,rightDescr=descriptor.compute(rightImg,rightKPTs)
        #         rdtime.seconds=time.time()-startTime
        #         ##pack into frame
        #         latestFrame.rdescr.append(cvb.cv2_to_imgmsg(rightDescr))
        #     processCount+=1
        #     print(str(processCount)+"/"+str(len(req.det_attrib)*len(req.desc_attrib)))
        fullOutputMessage.outputFrames.append(latestFrame)
    print("-----",imagesProcessed)
    imagesProcessed+=1
    return fullOutputMessage

if __name__ == '__main__':
    rospy.init_node("feature_node")
    setServices=[]
    s=rospy.Service("feature_node/singleImageDetection",singleImageDetection,singleImageDetection_fn)

    rospy.spin()