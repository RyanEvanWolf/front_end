#include <string>
 #include <unistd.h>
#include <front_end/nonfree.hpp>
#include <front_end/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <front_end/AKAZE.h>

#include <iostream>

#include <opencv2/highgui.hpp>


float getKeypoints(cv::Mat inImage,cv::FileNode detectorSettings,std::vector<cv::KeyPoint> &outKeyPoints);
bool castBool(std::string pythonBool);
#include <sys/time.h>

