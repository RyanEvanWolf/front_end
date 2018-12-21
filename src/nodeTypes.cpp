#include <front_end/nodeTypes.hpp>


Matrix<double,3,3> eulerToR(double roll_rad,double pitch_rad,double yaw_rad)
{
    Matrix<double,3,3> rx,ry,rz;

    rx=Rx(roll_rad);
    ry=Ry(pitch_rad);
    rz=Rz(yaw_rad);
    return rz*ry*rx;
}

Matrix<double,3,1> RToEuler(Matrix<double,3,3> inR)
{


    Matrix<double,3,1> outTheta;

    outTheta(1,0)=asin(inR(0,2));
    outTheta(0,0)=-acos(inR(0,0)/cos(outTheta(1,0)));
    outTheta(2,0)=-acos(inR(1,1)/cos(outTheta(1,0)));
    return outTheta;
}


	
Matrix<double,3,3> Rx(double rad)
{
    //roll
    Matrix<double,3,3> rx;
    rx<<    cos(rad),   -sin(rad),0,
            sin(rad),   cos(rad),0,
            0,0,1 ;
    return rx;   
}


Matrix<double,3,3> Ry(double rad)
{
    //pitch
    Matrix<double,3,3> ry;
	ry <<
               cos(rad),    0,      sin(rad),
               0,               1,      0,
               -sin(rad),   0,      cos(rad);
    return ry;
}
Matrix<double,3,3> Rz(double rad)
{
    //yaw
    Matrix<double,3,3> rz;
	rz <<1,0,0, 
         0,      cos(rad),    -sin(rad),
         0,      sin(rad),    cos(rad);
    return rz;
}

EulerPose::EulerPose()
{
    state.setZero();
    Pl= cv::Mat(3,4,CV_64F);
    Pr= cv::Mat(3,4,CV_64F);
}

Matrix<double,3,3> EulerPose::getR()
{
    return eulerToR(state(0,0),state(1,0),state(2,0));
}


Matrix<double,3,1> EulerPose::getT()    
{
    return state.block<3,1>(3,0);
}

Matrix<double,4,4> EulerPose::getH()
{
    Matrix<double,4,4> H;
    H.setZero();
    H(3,3)=1;
    H.block<3,3>(0,0)=getR();
    H.block<3,1>(0,3)=getT();
    return H;
}


