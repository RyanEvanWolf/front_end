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
		//thread info and mutexes for the image buffer
		//subscribes to an image topic, left and right, and attempts to
		//copy the image into a queue leftImages and rightImages respectively
		//------------------
		//condition_variable reduces the polling required by the thread by sleeping
		//when certain conditions are met. In this case, if the imageQueue is empty,
		//the thread sleeps until an image is pushed onto the queue
		std::queue<cv::Mat> leftImages,rightImages;
		boost::condition_variable leftImagesEmpty,rightImagesEmpty;
		boost::mutex mutexLImg,mutexRImg;

		//----------------------
		//image processing mutexes and buffers
		//each thread processes a single leftImages queue and extracts features according to  lDet and rDet
		std::queue<std::vector<cv::KeyPoint> > leftFeatures,rightFeatures;
		boost::condition_variable leftFeaturesEmpty,rightFeaturesEmpty;
		boost::mutex mutexLfeat,mutexRfeat;

		ros::NodeHandle n;
		image_transport::ImageTransport *it;
		image_transport::Subscriber leftSub;
		image_transport::Subscriber rightSub;
		void processLeftImage();
		void processRightImage();
		void processStereo();
		
		void BufferLeft(const sensor_msgs::ImageConstPtr& msg);
		void BufferRight(const sensor_msgs::ImageConstPtr& msg);
		//services
		bool updateDetector(front_end::setDetector::Request& req,front_end::setDetector::Response &res);
		//image processing
		cv::Ptr<cv::FeatureDetector> lDet,rDet;
		boost::mutex mutexlDet,mutexrDet;
		ros::ServiceServer detectorSrv;
	public:
		StereoRect cameraSettings_;
		StereoCamera(std::string cameraFile);
		~StereoCamera();

};
	
	
}

#endif
