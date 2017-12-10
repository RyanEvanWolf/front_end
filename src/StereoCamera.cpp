#include "front_end/StereoCamera.hpp"
namespace stereo
{

StereoCamera::StereoCamera(std::string cameraFile)
{
	cv::FileStorage in(cameraFile,cv::FileStorage::READ);
	in["StereoRect"]>>cameraSettings_;
	in.release();

	it= new image_transport::ImageTransport(n);

	leftSub=it->subscribe("bumblebee/leftROI", 5, &StereoCamera::BufferLeft, this);
	rightSub=it->subscribe("bumblebee/rightROI",5,&StereoCamera::BufferRight,this);
	detectorSrv=n.advertiseService("front_end/setDetector",&StereoCamera::updateDetector,this);

	boost::thread leftThread(boost::bind(&StereoCamera::processLeftImage,this));
	boost::thread rightThread(boost::bind(&StereoCamera::processRightImage,this));
}

StereoCamera::~StereoCamera()
{
	if(it!=NULL)
	{
		free(it);
	}
}

void StereoCamera::BufferLeft(const sensor_msgs::ImageConstPtr& msg)
{
	boost::mutex::scoped_lock lock(mutexLImg);
	bool const was_empty=leftImages.empty();
  leftImages.push((cv_bridge::toCvShare(msg, "8UC1")->image).clone());
  if(was_empty)
	{
		leftImagesEmpty.notify_one();
	}
}

void StereoCamera::BufferRight(const sensor_msgs::ImageConstPtr& msg)
{
	boost::mutex::scoped_lock lock(mutexRImg);
	bool const was_empty=rightImages.empty();
  rightImages.push((cv_bridge::toCvShare(msg, "8UC1")->image).clone());
  if(was_empty)
	{
		rightImagesEmpty.notify_one();
	}
}


void StereoCamera::processLeftImage()
{
	while(ros::ok())
	{
		boost::mutex::scoped_lock lock(mutexLImg);
    while(leftImages.empty())
    {
			leftImagesEmpty.wait(lock);
    }
		std::cout<<"received new image--"<<leftImages.size()<<std::endl;
		cv::Mat t;
		leftImages.front().copyTo(t);//.clone();
		leftImages.pop();
		std::cout<<"received new Size--"<<leftImages.size()<<std::endl;
		lock.unlock();
	}
}

void StereoCamera::processRightImage()
{
	while(ros::ok())
	{
		boost::mutex::scoped_lock lock(mutexRImg);
    while(rightImages.empty())
    {
			rightImagesEmpty.wait(lock);
    }
		std::cout<<"received new image--"<<rightImages.size()<<std::endl;
		cv::Mat t;
		rightImages.front().copyTo(t);//.clone();
		rightImages.pop();
		std::cout<<"received new Size--"<<rightImages.size()<<std::endl;
		lock.unlock();
	
	}
}



bool StereoCamera::updateDetector(front_end::setDetector::Request& req,front_end::setDetector::Response &res)
{
	return true;
}


/*
StereoCamera::StereoCamera(cv::Ptr<DetectorSettings> dl,cv::Ptr<DetectorSettings> dr,
					 cv::Ptr<DetectorSettings> del,cv::Ptr<DetectorSettings> der,
					 std::string stereoInputDir)
{
	ldet=dl;
	rdet=dr;
	ldesc=del;
	rdesc=der;
	
	cv::FileStorage c(stereoInputDir,cv::FileStorage::READ);
	if(!c.isOpened())
	{
		std::cerr<<"Stereo Configuration file not found\n";
	}
	c["StereoRect"]>>cameraSettings_;
	c.release();
	
	lundistort_=cv::Mat(cameraSettings_.L_fMapx_.size(),CV_8UC1);
	rundistort_=cv::Mat(cameraSettings_.R_fMapx_.size(),CV_8UC1);
	
	lroi_=lundistort_(cameraSettings_.l_ROI_);
	rroi_=rundistort_(cameraSettings_.r_ROI_);
}

void StereoCamera::extractStereoFrame(cv::Mat leftIn, cv::Mat rightIn, StereoFrame& outFrame)
{
	cv::remap(leftIn,lundistort_,cameraSettings_.L_fMapx_,cameraSettings_.L_fMapy_,cv::INTER_LINEAR);
	cv::remap(rightIn,rundistort_,cameraSettings_.R_fMapx_,cameraSettings_.R_fMapy_,cv::INTER_LINEAR);
	
	ldet->detect(lroi_,outFrame.leftFeatures_);
	rdet->detect(rroi_,outFrame.rightFeatures_);
	
	ldet->extract(lroi_,outFrame.leftFeatures_,outFrame.leftDescrip_);
	rdet->extract(rroi_,outFrame.rightFeatures_,outFrame.rightDescrip_);
	
	outFrame.inliersMask_.clear();
	for(int index=0;index<outFrame.leftFeatures_.size();index++)
	{
		outFrame.inliersMask_.push_back(1);
	}*/
	




	
	
}
