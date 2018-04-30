import numpy as np
from tf.transformations import quaternion_from_euler,quaternion_matrix,euler_from_matrix
from math import pi
import rosbag
import time

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


def getMotion(H):
    Result={}
    angles=euler_from_matrix(H[0:3,0:3],'szxy')
    Result["Roll"]=57.2958*angles[0]
    Result["Pitch"]=57.2958*angles[1]
    Result["Yaw"]=57.2958*angles[2]
    Result["X"]=H[0,3]
    Result["Y"]=H[1,3]
    Result["Z"]=H[2,3]
    return Result
