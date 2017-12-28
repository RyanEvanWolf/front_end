#include <ros/ros.h>
#include <front_end/WindowMatcher.hpp>
int main(int argc,char *argv[])
{
	ros::init(argc,argv,"front_end_window");
	WindowMatcher slidingWindow(3);
	ros::spin();
	
	return 0;
}
