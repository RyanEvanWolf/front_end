#!/usr/bin/env python
#####ros related imports
import rospy 
import cv2
import inspect

from front_end.utils import *
from front_end.srv import *

from cv_bridge import CvBridge

cvb=CvBridge()

def singleImageExtraction_fn(req):
    print("here")
    f=cvb.imgmsg_to_cv2(req.leftImg)
    return front_end.srv.singleImageExtractionResponse()

if __name__ == '__main__':
    rospy.init_node("feature_node")
    setServices=[]
    s=rospy.Service("feature_node/singleImageExtraction",singleImageExtraction,singleImageExtraction_fn)

    rospy.spin()