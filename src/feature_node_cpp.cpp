#include <ros/ros.h>
#include <string>

//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <front_end/utils.hpp>
#include <sys/time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <opencv2/highgui.hpp>

#include <front_end/singleImageDetection.h>



ros::ServiceServer singleImageDetectionSrv;
cv::FileStorage fs;
image_transport::ImageTransport *it;
	
bool fn_singleImageDetection(front_end::singleImageDetection::Request& req,front_end::singleImageDetection::Response &res)
{
	//get images from message
	cv::Mat left,right;
	left=(cv_bridge::toCvCopy(req.leftImg,"8UC1")->image);
	right=(cv_bridge::toCvCopy(req.rightImg,"8UC1")->image);
	std::vector<cv::KeyPoint> lKP,rKP;
	
	
	std::vector<std::string>::iterator it;  // declare an iterator to a vector of strings
	for(it = req.detID.begin(); it != req.detID.end(); it++)
	{
		front_end::frameDetection ans;
		ans.detID=(*it);
		front_end::ProcTime lTime,rTime;
		lTime.label="lKP";
		rTime.label="rKP";
		
		lTime.seconds=getKeypoints(left,fs[(*it)],lKP);
		rTime.seconds=getKeypoints(right,fs[(*it)],rKP);
		ans.processingTime.push_back(lTime);
		ans.processingTime.push_back(rTime);
		if(req.returnKP)
		{
			int kpCount=0;
			///////////
			//NOT IMPLEMENTED YET
		}
// 		cv::Mat displayL,displayR;
// 		drawKeypoints(left,lKP,displayL);
// 		drawKeypoints(right,rKP,displayR);
// 		cv::imshow("left",displayL);
// 		cv::imshow("right",displayR);
// 		cv::waitKey(1);
//for debugging purposes only
		std::cout<<(*it)<<":LEFT=";
		std::cout<<lTime.seconds<<"|"<<lKP.size();
		std::cout<<"|"<<rKP.size()<<std::endl;
        ans.nLeft= lKP.size();
        ans.nRight=rKP.size();
		res.outputFrames.push_back(ans);
	}
	return true;
}

int main(int argc,char *argv[])
{	

	cv::initModule_nonfree();
	std::cout<<"OpenCV version : " << CV_VERSION << std::endl;
	std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
	std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
	std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;
	std::string nodeName="feature_node_cpp";
    ros::init(argc,argv,nodeName);
    ros::NodeHandle n;
    new image_transport::ImageTransport(n);
	fs.open("/home/ubuntu/detectorLookupTable.yaml", cv::FileStorage::READ);
	std::cout<<"Loaded Detector table"<<std::endl;
	
	singleImageDetectionSrv=n.advertiseService(nodeName+"/singleImageDetection",fn_singleImageDetection);
	
	/*cv::FileNode fn = fs.root();
	for (cv::FileNodeIterator fit = fn.begin(); fit != fn.end(); ++fit)
	{
		cv::FileNode item = *fit;
		std::string somekey = item.name();
		std::cout << somekey <<","<<(std::string)item["Name"]<<":";
		
		
		
		std::cout<<(std::string)item["Param"][0]<< std::endl;
	}*/

	std::cout<<"Spinning"<<std::endl;
    ros::spin();
    return 0;
}
