#include <ros/ros.h>
#include <string>



#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"





#include <sys/time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>



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

///////////////
//DETECTOR PARAMETERS
////////////////
int gridRow=2;
int gridCol=3;
int setPoint=3000;
int minthreshold=4;
int gridSetpoint=0;
cv::Mat lThresholds,rThresholds;
int gridHeight,gridWidth;


sensor_msgs::RegionOfInterest roiSettings;
cv::Rect lroi;



ros::Publisher debugFeatures,debugMatches;
ros::Publisher debugTimeFeatures,debugTimeMatches,debugTimeDescriptor;
ros::ServiceServer controlServer;


void updateSetPoint(int newSetPoint)
{
    setPoint=newSetPoint;
    gridSetpoint=(int)(((float)setPoint)/((float)gridRow*gridCol));
    std::cout<<"new detector Settings\n";
    std::cout<<"setPoint:"<<setPoint<<",gridSetPoint:"<<gridSetpoint<<std::endl;
}

void setDetectorThresholds(int threshold)
{
    cv::Mat temp;
    temp=threshold*cv::Mat::ones(gridRow,gridCol,CV_32SC1);
    temp.copyTo(lThresholds);
    temp.copyTo(rThresholds);
   
    std::cout<<"new Thresholds\n----------\n";
    std::cout<<lThresholds<<rThresholds<<"\n-----------\n";

}

bool fn_controlDetection(front_end::controlDetection::Request &req,
													front_end::controlDetection::Response &res)
{
	
    updateSetPoint(req.setPoint);
    
	res.newSetPoint=setPoint;
    setDetectorThresholds(req.threshold);
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
	updateSetPoint(1000);
    setDetectorThresholds(15);
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
    gridHeight=(int)(((float)lroi.height)/((float)gridRow));
    gridWidth=(int)(((float)lroi.width)/((float)gridCol));
    std::cout<<"Grid Height:"<<gridHeight<<",Grid Width:"<<gridWidth<<std::endl;

	rightSub=it->subscribe(rightImageTopic,5,BufferRight);//,this);
  leftSub=it->subscribe(leftImageTopic,5,BufferLeft);//,this);
	debugFeatures=n.advertise<std_msgs::Float32>("stereo/debug/detection",2);
  debugMatches=n.advertise<std_msgs::Float32>("stereo/debug/matches",2);
	debugTimeFeatures=n.advertise<std_msgs::Float32>("stereo/time/detection",2);
	debugTimeMatches=n.advertise<std_msgs::Float32>("stereo/time/matches",2);
    debugTimeDescriptor=n.advertise<std_msgs::Float32>("stereo/time/description",2);
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
	
  rightImages.push(image.clone());
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
	
  leftImages.push(image.clone());
  if(was_empty)
	{
		leftImagesEmpty.notify_one();
	}
	std::cout<<"lQ:"<<leftImages.size()<<std::endl;
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
  
int clip(int inValue,int min,int max)
{
    int ans=inValue;
    if(inValue<min)
    {
        ans=min;
    }
    if(inValue>max)
    {
        ans=max;
    }
    return ans;
}  

void stereoMatch()
{
	cv::namedWindow("out",cv::WINDOW_NORMAL);
	cv::Mat currentLeft,currentRight;
    cv::Mat currentROIl,currentROIr;
	cv::Mat lDescriptor,rDescriptor;
	std_msgs::Float32 debugOutMsg;
    
    /////////////////
    //change descriptor types here
    /////////////////////////
	cv::BriefDescriptorExtractor extractor(16);
    
	cv::BFMatcher m(cv::NORM_HAMMING,true);
	cv::Size winSize = cv::Size( 5, 5 );
    cv::Size zeroZone = cv::Size( -1, -1 );
    cv::TermCriteria criteria = cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

  struct timeval  tv1, tv2;
	while(ros::ok())
	{
		//Wait for images to be published
		std::vector<cv::KeyPoint> leftKP,rightKP;

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
        
        currentROIl=currentLeft(lroi);
        currentROIr=currentRight(lroi);
        
        
		gettimeofday(&tv1, NULL);
        for(int row=0;row<gridRow;row++)
        {
            for(int col=0;col<gridCol;col++)
            {
                /////
                //get left features
                cv::Rect gridROI;
       
                gridROI.x=col*gridWidth;
                gridROI.y=row*gridHeight;
                gridROI.width=gridWidth;
                gridROI.height=gridHeight;
                cv::Mat lgridImg=currentROIl(gridROI);
                cv::Mat rgridImg=currentROIr(gridROI);

                std::vector<cv::KeyPoint> rawDetectionsLeft,rawDetectionsRight;
                cv::FASTX(lgridImg,rawDetectionsLeft,lThresholds.at<int>(row,col),true,cv::FastFeatureDetector::TYPE_7_12);
                int error=rawDetectionsLeft.size()-gridSetpoint;
                if(abs(error)>0.2*gridSetpoint)
                {
                    if(error>0)
                    {
                        lThresholds.at<int>(row,col)=clip(lThresholds.at<int>(row,col)+1,minthreshold,80);
                    }
                    else
                    {
                        lThresholds.at<int>(row,col)=clip(lThresholds.at<int>(row,col)-1,minthreshold,80);
                    } 
                }
                cv::FASTX(rgridImg,rawDetectionsRight,rThresholds.at<int>(row,col),true,cv::FastFeatureDetector::TYPE_7_12);
                error=rawDetectionsRight.size()-gridSetpoint;
                if(abs(error)>0.2*gridSetpoint)
                {
                    if(error>0)
                    {
                        rThresholds.at<int>(row,col)=clip(rThresholds.at<int>(row,col)+1,minthreshold,80);
                    }
                    else
                    {
                        rThresholds.at<int>(row,col)=clip(rThresholds.at<int>(row,col)-1,minthreshold,80);
                    } 
                } 
                /////////////////////
                //subPixel refinement
                for(int kpIndex=0;kpIndex<rawDetectionsLeft.size();kpIndex++)
                {
                    std::vector<cv::Point2f> refinedPointL;
                   
                    refinedPointL.push_back(rawDetectionsLeft.at(kpIndex).pt);
                    cv::cornerSubPix(lgridImg,refinedPointL, winSize, zeroZone, criteria );
                    rawDetectionsLeft.at(kpIndex).pt=refinedPointL.at(0); 
                }
                       
                for(int kpIndex=0;kpIndex<rawDetectionsRight.size();kpIndex++)
                {
                    std::vector<cv::Point2f> refinedPointR;
                   
                    refinedPointR.push_back(rawDetectionsRight.at(kpIndex).pt);
                    cv::cornerSubPix(rgridImg,refinedPointR, winSize, zeroZone, criteria ); 
                    rawDetectionsRight.at(kpIndex).pt=refinedPointR.at(0); 
                }
                //////////////
                //add offsets;
                for(int kpIndex=0;kpIndex<rawDetectionsLeft.size();kpIndex++)
                {
                    rawDetectionsLeft.at(kpIndex).pt.x= rawDetectionsLeft.at(kpIndex).pt.x+gridROI.x+lroi.x;
                    rawDetectionsLeft.at(kpIndex).pt.y=rawDetectionsLeft.at(kpIndex).pt.y+gridROI.y+lroi.y;
                    
                }
                for(int kpIndex=0;kpIndex<rawDetectionsRight.size();kpIndex++)
                {
                    rawDetectionsRight.at(kpIndex).pt.x= rawDetectionsRight.at(kpIndex).pt.x+gridROI.x+lroi.x;
                    rawDetectionsRight.at(kpIndex).pt.y=rawDetectionsRight.at(kpIndex).pt.y+gridROI.y+lroi.y;
                }
                leftKP.insert(leftKP.end(),rawDetectionsLeft.begin(),rawDetectionsLeft.end());
                rightKP.insert(rightKP.end(),rawDetectionsRight.begin(),rawDetectionsRight.end());
            }
        }
        gettimeofday(&tv2, NULL);
        float executionSeconds=((float)(tv2.tv_usec - tv1.tv_usec) / 1000000)+((float)(tv2.tv_sec - tv1.tv_sec));
        
        gettimeofday(&tv1, NULL);
		extractor.compute(currentLeft,leftKP,lDescriptor);
		extractor.compute(currentRight,rightKP,rDescriptor);
        gettimeofday(&tv2, NULL);
		float descTime=((float)(tv2.tv_usec - tv1.tv_usec) / 1000000)+((float)(tv2.tv_sec - tv1.tv_sec));
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
			if(abs(epiDistance)<=0.7)
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

		debugOutMsg.data=descTime;
		debugTimeDescriptor.publish(debugOutMsg);
        
        
		cv::Mat outImg;
        cv::drawMatches(currentLeft,leftKP,currentRight,rightKP,goodMatch,outImg);
	//	cv::drawKeypoints(currentLeft,leftKP,currentLeft);// const Scalar& color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT )
//~ 
		cv::imshow("out",outImg);
		cv::waitKey(1);
	}

}




