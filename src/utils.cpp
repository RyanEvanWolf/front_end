
#include <front_end/utils.hpp>

bool castBool(std::string pythonBool)
{
    if(pythonBool=="False")
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

float getKeypoints(cv::Mat inImage,cv::FileNode detectorSettings,std::vector<cv::KeyPoint> &outKeyPoints)
{
    //returns the time taken in seconds
    std::string detectorName=detectorSettings["Name"];
    if(detectorName=="FAST")
    {
        //build the detector
        int thresh,type;
        bool supp;
        thresh=std::stoi((std::string)detectorSettings["Param"][0]);
        type=std::stoi((std::string)detectorSettings["Param"][1]);
        supp=castBool((std::string)detectorSettings["Param"][2]);
        struct timeval  tv1, tv2;
        gettimeofday(&tv1, NULL);
        cv::FASTX(inImage,outKeyPoints,thresh,supp,type);
        gettimeofday(&tv2, NULL);
        float executionSeconds=((float)(tv2.tv_usec - tv1.tv_usec) / 1000000)+((float)(tv2.tv_sec - tv1.tv_sec));
        return executionSeconds;
    }
    else
    {
        return 0;
    }
}

