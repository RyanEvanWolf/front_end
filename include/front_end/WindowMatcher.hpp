#ifndef WINDOW_MATCHER_HEADERS_HPP
#define WINDOW_MATCHER_HEADERS_HPP


#include <ros/ros.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>


//services includes
#include <bumblebee/getOffset.h>
#include <front_end/setDetector.h>
#include <bumblebee/getQ.h>
//messages includes
#include <front_end/Feature.h>
#include <front_end/StereoFrame.h>
#include <front_end/FrameTracks.h>
#include <front_end/Window.h>
#include <front_end/WindowFrame.h>
#include <front_end/Landmark.h>
#include <front_end/InterWindowFrame.h>
#include <sensor_msgs/Image.h>



#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <queue>
#include <list>

#include <std_msgs/Int8.h>
#include <std_msgs/String.h>
//motion extraction
#include <five-point-nister/five-point.hpp>
#include <Structures/Transforms/Isometry.hpp>



#include <chrono>

class WindowMatcher
{
	private:
		int nWindow;
		ros::Subscriber stereoSub;
		ros::Subscriber normSub;
		ros::Subscriber encodingSub;
		ros::Publisher windowPub;
		ros::Publisher statePub;
		ros::Publisher leftTracks,rightTracks;
		ros::ServiceClient getQmapClient;
		ros::NodeHandle n;
		void newStereo(const front_end::StereoFrame::ConstPtr& msg);
		std::vector<front_end::InterWindowFrame> interFrame;
		std::vector<front_end::WindowFrame> window;
		cv::Mat Q;
		cv::Rect searchRegion;
		image_transport::ImageTransport *it;
		std_msgs::Int8 normType;
		std::string encodingType;
		void updateNorm(const std_msgs::Int8::ConstPtr& msg);
		void updateEncoding(const std_msgs::String::ConstPtr& msg);
		std::vector<cv::DMatch> loweRejection(std::vector< std::vector<cv::DMatch> > initial);
		front_end::FrameTracks convertToMessage(std::vector<front_end::StereoMatch> currentMatches,
																													std::vector<front_end::StereoMatch> previousMatches,	
																													std::vector<cv::DMatch>);
		front_end::FrameTracks extractMotion(std::vector<front_end::StereoMatch> currentMatches,
																													std::vector<front_end::StereoMatch> previousMatches,	
																													std::vector<cv::DMatch> matches);
		void publishCurrentState();
		void triangulate(front_end::Landmark &in);
		float loweRejectionRatio=0.8;
		int debug;
	public:
		WindowMatcher(int windowSize);
};


#endif
