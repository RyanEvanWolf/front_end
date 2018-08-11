#include <string>
 #include <unistd.h>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>


#include <iostream>

#include <opencv2/highgui.hpp>


float getKeypoints(cv::Mat inImage,cv::FileNode detectorSettings,std::vector<cv::KeyPoint> &outKeyPoints);
bool castBool(std::string pythonBool);
#include <sys/time.h>

