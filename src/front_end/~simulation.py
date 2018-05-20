import cv2
import math
import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_from_euler, quaternion_matrix

def dominantTranslation(zBase=0.1,noise=0.1):
        
        frame={}
        frame["X"]=np.random.normal(0,noise,1)
        frame["Z"]=np.random.normal(zBase,noise,1)
        frame["Y"]=np.random.normal(0,noise,1)
        t=np.zeros((3,1),dtype=np.float64)
        t[0,0]=frame["X"]
        t[1,0]=frame["Y"]
        t[2,0]=frame["Z"]756-85
        frame["T"]=t
        return frame

def noisyRotation(noise=0.5):
        frame={}
        r=math.radians(np.random.normal(0,noise,1))
        frame["Roll"]=r
        p=math.radians(np.random.normal(0,noise,1))
        frame["Pitch"]=p
        y=math.radians(np.random.normal(0,noise,1))
        frame["Yaw"]=y
        q=quaternion_from_euler(r,p,y,'szyx')
        frame["R"]=quaternion_matrix(q)[0:3,0:3]
        return frame

class simulatedStereoCamera:
    def __init__(self,linfo,rinfo,Q):
        self.Pl=np.zeros((3,4),dtype=np.float64)
        self.Pr=np.zeros((3,4),dtype=np.float64)
        self.Q=Q
        for row in range(0,3):
        for col in range(0,4):
                pl[row,col]=leftInfo.P[row*4+col]
        pr[row,col]=rightInfo.P[row*4+]
