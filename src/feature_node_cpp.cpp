#include <ros/ros.h>


#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>


#include <iostream>

#include <opencv2/highgui.hpp>
int main(int argc,char *argv[])
{
	ros::init(argc,argv,"front_end");

    cv::Mat img =cv::imread("/home/ryan/DATA/A/00500.ppm",CV_LOAD_IMAGE_GRAYSCALE);
    std::vector<cv::KeyPoint> out,out2;
    cv::Mat eO;
    cv::Ptr<cv::FastFeatureDetector> det=cv::FastFeatureDetector::create();
    
    det->detect(img,out);
    out2.push_back(out.at(0));
    out2.push_back(out.at(1));
   //     out2.push_back(out.at(0));
   // out2.push_back(out.at(0));
   //     out2.push_back(out.at(0));
   // out2.push_back(out.at(0));
   //     out2.push_back(out.at(0));
   // out2.push_back(out.at(0));

    std::cout<<out2.at(0).pt<<","<<out2.size()<<","<<out.size()<<std::endl;
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(64,false);
    extractor->compute(img,out,eO);

    std::cout<<eO.size()<<std::endl;
    cv::imshow("A",img);
    cv::waitKey(0);

    cv::destroyAllWindows();
	ros::spin();
	
	return 0;
}
