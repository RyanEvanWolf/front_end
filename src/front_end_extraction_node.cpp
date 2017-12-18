#include <ros/ros.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <front_end/setDetector.h>
#include <front_end/detectCurrent.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <opencv2/highgui.hpp>

#include <chrono>
cv::Ptr<cv::Feature2D> Det;
image_transport::ImageTransport *it;
image_transport::Subscriber leftSub;
cv::Mat leftImage;

bool updateDetector(front_end::setDetector::Request& req,front_end::setDetector::Response &res)
{
	//built for opencv 3 functions
	std::string name=(req.Name.data);
	//set the feature detector
	if(name=="ORB")	
	{
		int maxfeatures,level,edge,wta,score,patch;
		float scale;

		Det=cv::ORB::create(static_cast<int>(req.orbConfig.maxFeatures.data),
							static_cast<float>(req.orbConfig.scale.data),
							static_cast<int>(req.orbConfig.level.data),
							static_cast<int>(req.orbConfig.edge.data),
							0,
							static_cast<int>(req.orbConfig.wta.data),
							static_cast<int>(req.orbConfig.score.data),
							static_cast<int>(req.orbConfig.patch.data));
		//Det=cv::FastFeatureDetector::create();
		
	}

	return true;
}

bool detect(front_end::detectCurrent::Request& req,front_end::detectCurrent::Response &res)
{
	std::vector<cv::KeyPoint> output;
	auto start=std::chrono::steady_clock::now();
	Det->detect(leftImage,output);
	auto end=std::chrono::steady_clock::now();
	res.Time.data=std::chrono::duration<double,std::milli>(end-start).count();
	res.nleft.data=output.size();
	std::cout<<output.size()<<std::endl;
	return true;
}

void update(const sensor_msgs::ImageConstPtr& msg)
{
	(cv_bridge::toCvShare(msg, "8UC1")->image).copyTo(leftImage);
}


int main(int argc,char *argv[])
{
	ros::init(argc,argv,"front_end_extraction");
	ros::NodeHandle n;

	it= new image_transport::ImageTransport(n);
	image_transport::Subscriber sub = it->subscribe("/bumblebee/leftROI", 5, update);
	ros::ServiceServer detectorSrv=n.advertiseService("front_end/setExtractorDetector",updateDetector);
	ros::ServiceServer detSrv=n.advertiseService("front_end/detectCurrent",detect);
	
	
	
	ros::spin();
	
	return 0;
} 
