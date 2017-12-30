#ifndef WINDOW_MATCHER_HEADERS_HPP
#define WINDOW_MATCHER_HEADERS_HPP


#include <ros/ros.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include <front_end/setDetector.h>
#include <front_end/Feature.h>
#include <front_end/StereoFrame.h>
#include <front_end/FrameTracks.h>
#include <front_end/Window.h>

#include <bumblebee/getOffset.h>


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
		ros::NodeHandle n;
		void newStereo(const front_end::StereoFrame::ConstPtr& msg);
		std::list<std::vector<front_end::StereoMatch> > windowData;
		std::list<front_end::FrameTracks> motionData;
		cv::Rect searchRegion;
		image_transport::ImageTransport *it;
		std_msgs::Int8 normType;
		std::string encodingType;
		void updateNorm(const std_msgs::Int8::ConstPtr& msg);
		void updateEncoding(const std_msgs::String::ConstPtr& msg);
		cv::Mat getSearchMask(std::vector<front_end::StereoMatch> currentMatches,std::vector<front_end::StereoMatch> previousMatches);
		std::vector< std::vector<cv::DMatch> > knnWindowMatch(std::vector<front_end::StereoMatch> currentMatches,
																													std::vector<front_end::StereoMatch> previousMatches,	
																													cv::Mat mask);
		std::vector<cv::DMatch> loweRejection(std::vector< std::vector<cv::DMatch> > initial);
		front_end::FrameTracks convertToMessage(std::vector<front_end::StereoMatch> currentMatches,
																													std::vector<front_end::StereoMatch> previousMatches,	
																													std::vector<cv::DMatch>);
		front_end::FrameTracks extractMotion(std::vector<front_end::StereoMatch> currentMatches,
																													std::vector<front_end::StereoMatch> previousMatches,	
																													std::vector<cv::DMatch> matches);
		void publishCurrentState();
	public:
		WindowMatcher(int windowSize);
};


#endif
