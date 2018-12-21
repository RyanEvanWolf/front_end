#ifndef FRONT_END_NODE_TYPES_G2O_HEADER
#define FRONT_END_NODE_TYPES_G2O_HEADER

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

////////////////////////////
#include <Eigen/StdVector>

#include <unordered_set>

#include <iostream>
#include <stdint.h>

using namespace Eigen;
using namespace std;

Matrix<double,3,3> eulerToR(double roll_rad,double pitch_rad,double yaw_rad);
Matrix<double,3,3> Rx(double rad);
Matrix<double,3,3> Ry(double rad);
Matrix<double,3,3> Rz(double rad);
Matrix<double,3,1> RToEuler(Matrix<double,3,3> inR);

class EulerPose
{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		EulerPose();
        EulerPose(Matrix<double,6,1> initialX);
        Matrix<double,6,1> state; //roll,pitch,yaw,x,y,z ,,, radians meters
        Matrix<double,3,3> getR();
        Matrix<double,3,1> getT();
        Matrix<double,3,1> getC();
        Matrix<double,4,4> getH();
        cv::Mat Pl,Pr;
};


// class StereoEuler : public BaseVertex<6,EulerPose>
// {
//    public:
// 		EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
//         StereoEuler();
// }

#endif