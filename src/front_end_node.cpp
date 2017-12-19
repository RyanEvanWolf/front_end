#include <ros/ros.h>




#include <front_end/StereoCamera.hpp>
#define DEFAULT_RECTIFICATION_FILE "/home/ubuntu/ConfigurationFiles/stereo_ParameterFive.xml"


int main(int argc,char *argv[])
{
	ros::init(argc,argv,"front_end");
	stereo::StereoCamera bumble;
	ros::spin();
	
	return 0;
}
