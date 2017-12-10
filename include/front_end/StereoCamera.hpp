#ifndef STEREO_CAMERA_HEADERS_HPP
#define STEREO_CAMERA_HEADERS_HPP


#include "Structures/CameraInfo/StereoRect.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include <front_end/setDetector.h>

#include <queue>

namespace stereo 
{
	
class StereoCamera 
{
	private:
		std::queue<cv::Mat> leftImages,rightImages;
		boost::condition_variable leftImagesEmpty,rightImagesEmpty;
		boost::mutex mutexLImg,mutexRImg;
		ros::NodeHandle n;
		image_transport::ImageTransport *it;//(&n);
		image_transport::Subscriber leftSub;
		image_transport::Subscriber rightSub;
		void processLeftImage();
		void processRightImage();
		void BufferLeft(const sensor_msgs::ImageConstPtr& msg);
		void BufferRight(const sensor_msgs::ImageConstPtr& msg);
		//services
		bool updateDetector(front_end::setDetector::Request& req,front_end::setDetector::Response &res);
		ros::ServiceServer detectorSrv;
	public:
		StereoRect cameraSettings_;
		cv::Mat lundistort_,rundistort_;
		cv::Mat lroi_,rroi_;
		//cv::Ptr<cv::Feature2d> detector;
		StereoCamera(std::string cameraFile);
		~StereoCamera();


		//StereoCamera(cv::Ptr<DetectorSettings> dl,cv::Ptr<DetectorSettings> dr,
		//			 cv::Ptr<DetectorSettings> del,cv::Ptr<DetectorSettings> der,
		//			 std::string stereoInputDir);
		//void extractStereoFrame(cv::Mat leftIn,cv::Mat rightIn,StereoFrame &outFrame);
};
	
	
}

#endif
