#include <ros/ros.h>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <sys/time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <opencv2/highgui.hpp>


#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32.h>


#include <front_end/singleImageDetection.h>
#include <front_end/controlDetection.h>

#include <front_end/nonfree.hpp>
#include <front_end/features2d.hpp>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <queue>
#include <unistd.h>

#include <algorithm>


//////////////////////
//topic names
/////////////////////



std::string nodeName="detect_node_cpp";
std::string leftImageTopic="Dataset/left";
std::string rightImageTopic="Dataset/right";
std::string roiTopic="bumblebee_configuration/Left/subROI/CameraInfo";


std::queue<cv::Mat> leftImages,rightImages;

image_transport::ImageTransport *it;
image_transport::Subscriber rightSub,leftSub;
boost::condition_variable leftImagesEmpty,rightImagesEmpty;
boost::mutex mutexLImg,mutexRImg;






int detectionWindow[5]={8,9,10,11,12};
int setPoint=3000;
int maxthreshold=60;
int minthreshold=3;


sensor_msgs::RegionOfInterest roiSettings;
cv::Rect lroi;



ros::Publisher debugFeatures,debugMatches;
ros::Publisher debugTimeFeatures,debugTimeMatches;
ros::ServiceServer controlServer;



bool fn_controlDetection(front_end::controlDetection::Request &req,
													front_end::controlDetection::Response &res)
{
	
	res.previousThreshold=detectionWindow[2];


	detectionWindow[2]=req.threshold;
	setPoint=req.setPoint;
	res.newSetPoint=setPoint;
	return true;

	
}

void BufferRight(const sensor_msgs::ImageConstPtr& msg);//on image publish, push to the queue
void BufferLeft(const sensor_msgs::ImageConstPtr& msg);
void stereoMatch();






int main(int argc,char *argv[])
{	

	cv::initModule_nonfree();
	std::cout<<"OpenCV version : " << CV_VERSION << std::endl;
	std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
	std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
	std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;
	
	ros::init(argc,argv,nodeName);
	
	ros::NodeHandle n;
	it=new image_transport::ImageTransport(n);
	controlServer=n.advertiseService("stereo/control/detection",fn_controlDetection);	
	std::cout<<"waiting for region of interest settings "<<roiTopic<<std::endl;


	boost::shared_ptr<sensor_msgs::CameraInfo const> setting;
	setting=ros::topic::waitForMessage<sensor_msgs::CameraInfo>(roiTopic,n);
	roiSettings=setting->roi;
	lroi.x=setting->roi.x_offset;
	lroi.y=setting->roi.y_offset;
	lroi.width=setting->roi.width;
	lroi.height=setting->roi.height;
	std::cout<<"set ROI "<<lroi<<std::endl;


	rightSub=it->subscribe(rightImageTopic,5,BufferRight);//,this);
  leftSub=it->subscribe(leftImageTopic,5,BufferLeft);//,this);
	debugFeatures=n.advertise<std_msgs::Float32>("stereo/debug/detection",2);
  debugMatches=n.advertise<std_msgs::Float32>("stereo/debug/matching",2);
	debugTimeFeatures=n.advertise<std_msgs::Float32>("stereo/time/detection",2);
	debugTimeMatches=n.advertise<std_msgs::Float32>("stereo/time/matching",2);
	boost::thread stereoThread(stereoMatch);//boost::bind(&StereoCamera::processStereo,this));

	std::cout<<"Spinning"<<std::endl;
	ros::spin();
    return 0;
}


void BufferRight(const sensor_msgs::ImageConstPtr& msg)
{
	boost::mutex::scoped_lock lock(mutexRImg);
	bool const was_empty=rightImages.empty();
	cv::Mat image=cv_bridge::toCvShare(msg, "8UC1")->image;
	cv::Mat roiImage=image(lroi);
  rightImages.push(roiImage.clone());
  if(was_empty)
	{
		rightImagesEmpty.notify_one();
	}
	std::cout<<"rQ:"<<rightImages.size()<<std::endl;
}

void BufferLeft(const sensor_msgs::ImageConstPtr& msg)
{
	boost::mutex::scoped_lock lock(mutexLImg);
	bool const was_empty=leftImages.empty();
	cv::Mat image=cv_bridge::toCvShare(msg, "8UC1")->image;
	cv::Mat roiImage=image(lroi);
  leftImages.push(roiImage.clone());
  if(was_empty)
	{
		leftImagesEmpty.notify_one();
	}
	std::cout<<"lQ:"<<leftImages.size()<<std::endl;
}


void updateDetectionWindow()
{

	detectionWindow[0]=detectionWindow[2]-2 > minthreshold ?detectionWindow[2]-2 :minthreshold;
	detectionWindow[1]=detectionWindow[2]-1 > minthreshold ?detectionWindow[2]-1:minthreshold;
	detectionWindow[3]=detectionWindow[2]+1;
	detectionWindow[4]=detectionWindow[2]+2;


}


int min_index(int *a, int n)
  {
      if(n <= 0) return -1;
      int i, min_i = 0;
      int min = a[0];
      for(i = 1; i < n; ++i){
          if(a[i] < min){
              min = a[i];
              min_i = i;
          }
      }
      return min_i;
  }

void stereoMatch()
{
	cv::namedWindow("out",cv::WINDOW_NORMAL);
	cv::Mat currentLeft,currentRight;
	cv::Mat lDescriptor,rDescriptor;
	std_msgs::Float32 debugOutMsg;
	cv::BriefDescriptorExtractor extractor(16);


	cv::BFMatcher m(cv::NORM_HAMMING,true);
	
		

//BriefDescriptorExtractor::create( int bytes, bool use_orientation )
  struct timeval  tv1, tv2;
	while(ros::ok())
	{
		//Wait for images to be published
		std::vector<cv::KeyPoint> coarseDetections[5],leftKP,rightKP;



		int coarseTotals[5];
		boost::mutex::scoped_lock lockLf(mutexLImg);
		while(leftImages.empty())
		{
			leftImagesEmpty.wait(lockLf);
		}

		leftImages.front().copyTo(currentLeft);
		leftImages.pop();		
		lockLf.unlock();

		boost::mutex::scoped_lock lockRf(mutexRImg);
		while(rightImages.empty())
		{
			rightImagesEmpty.wait(lockRf);
		}

		rightImages.front().copyTo(currentRight);
		rightImages.pop();		
		lockRf.unlock();	
		gettimeofday(&tv1, NULL);
		updateDetectionWindow();
		std::cout<<"minWindow";
		int a=0;
		for(a=0;a<5;a++)
		{
			std::cout<<","<<detectionWindow[a];
		}
		std::cout<<std::endl;

		for(int detectionIndex=0;detectionIndex<5;detectionIndex++)
		{
			cv::FASTX(currentLeft,coarseDetections[detectionIndex],
								detectionWindow[detectionIndex],
								true,
								cv::FastFeatureDetector::TYPE_7_12);
			coarseTotals[detectionIndex]=abs(coarseDetections[detectionIndex].size()-setPoint);
		}
		
		int bestfeaturesIndex=min_index(coarseTotals,5);
		leftKP=coarseDetections[bestfeaturesIndex];
		
			cv::FASTX(currentRight,rightKP,
								detectionWindow[bestfeaturesIndex],
								true,
								cv::FastFeatureDetector::TYPE_7_12);
		
		detectionWindow[2]=detectionWindow[bestfeaturesIndex];
		std::cout<<"threshold:"<<detectionWindow[2];
		extractor.compute(currentLeft,leftKP,lDescriptor);
		extractor.compute(currentRight,rightKP,rDescriptor);

		gettimeofday(&tv2, NULL);
		float executionSeconds=((float)(tv2.tv_usec - tv1.tv_usec) / 1000000)+((float)(tv2.tv_sec - tv1.tv_sec));
		//////////////////
		//matching
		gettimeofday(&tv1, NULL);
		std::vector<cv::DMatch> initialMatches;		
		m.match(lDescriptor,rDescriptor,initialMatches);

		std::vector<int> inlierIndexes;
		std::vector<cv::DMatch> goodMatch;

		for(int matchIndex=0;matchIndex<initialMatches.size();matchIndex++)
		{
			float epiDistance=leftKP.at(initialMatches.at(matchIndex).queryIdx).pt.y-rightKP.at(initialMatches.at(matchIndex).trainIdx).pt.y;
			if(abs(epiDistance)<1)
			{
				inlierIndexes.push_back(matchIndex);
				goodMatch.push_back(initialMatches.at(matchIndex));
			}
		}
		gettimeofday(&tv2, NULL);
		float MatchSeconds=((float)(tv2.tv_usec - tv1.tv_usec) / 1000000)+((float)(tv2.tv_sec - tv1.tv_sec));

		debugOutMsg.data=leftKP.size();
		debugFeatures.publish(debugOutMsg);

		debugOutMsg.data=executionSeconds;
		debugTimeFeatures.publish(debugOutMsg);


		debugOutMsg.data=goodMatch.size();
		debugMatches.publish(debugOutMsg);

		debugOutMsg.data=MatchSeconds;
		debugTimeMatches.publish(debugOutMsg);
		cv::Mat outImg;
    cv::drawMatches(currentLeft,leftKP,currentRight,rightKP,goodMatch,outImg);

		

		cv::drawKeypoints(currentLeft,leftKP,currentLeft);// const Scalar& color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT )

		cv::imshow("out",outImg);
		cv::waitKey(1);
	}

}




