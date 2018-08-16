
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
    if(detectorName=="SURF")
    {
        
        
        double thresh=std::stod((std::string)detectorSettings["Param"][0]);
        int octaves,octavesLayers;
        bool extended,upright;
        octaves=std::stoi((std::string)detectorSettings["Param"][1]);
        octavesLayers=std::stoi((std::string)detectorSettings["Param"][2]);
        extended=castBool((std::string)detectorSettings["Param"][3]);
        upright=castBool((std::string)detectorSettings["Param"][4]);

        cv::SURF det=cv::SURF(thresh,octaves,octavesLayers,extended,upright);
        struct timeval  tv1, tv2;
        gettimeofday(&tv1, NULL);
        det.detect(inImage,outKeyPoints);
        gettimeofday(&tv2, NULL);
        float executionSeconds=((float)(tv2.tv_usec - tv1.tv_usec) / 1000000)+((float)(tv2.tv_sec - tv1.tv_sec));
        return executionSeconds;
    }
    if(detectorName=="AKAZE")
    {
        AKAZEOptions options;
        return 0;
    }
    if(detectorName=="BRISK")
    {
        int thresh,octaves;
        float scale;
        
        thresh=std::stoi((std::string)detectorSettings["Param"][0]);
        octaves=std::stoi((std::string)detectorSettings["Param"][1]);
        scale=std::stof((std::string)detectorSettings["Param"][2]);
        //RISK::BRISK(int thresh=30, int octaves=3, float patternScale=1.0f)
        cv::BRISK det=cv::BRISK(thresh,octaves,scale);
        struct timeval  tv1, tv2;
        gettimeofday(&tv1, NULL);
        det.detect(inImage,outKeyPoints);
        gettimeofday(&tv2, NULL);
        float executionSeconds=((float)(tv2.tv_usec - tv1.tv_usec) / 1000000)+((float)(tv2.tv_sec - tv1.tv_sec));
        return executionSeconds;
    }
    
    if(detectorName=="ORB")
    {
        int nfeat,nlevel,edge,wta,score,patch,fast;
        float scale;
        scale=std::stof((std::string)detectorSettings["Param"][0]);
        nlevel=std::stoi((std::string)detectorSettings["Param"][1]);
        edge=std::stoi((std::string)detectorSettings["Param"][2]);
        wta=std::stoi((std::string)detectorSettings["Param"][3]);
        score=std::stoi((std::string)detectorSettings["Param"][4]);
        patch=std::stoi((std::string)detectorSettings["Param"][5]);
        fast=std::stoi((std::string)detectorSettings["Param"][6]);

        cv::ORB det=cv::ORB(25000,scale,nlevel,edge,0,wta,score,patch);

        struct timeval  tv1, tv2;
        gettimeofday(&tv1, NULL);
        det.detect(inImage,outKeyPoints);
        gettimeofday(&tv2, NULL);
        float executionSeconds=((float)(tv2.tv_usec - tv1.tv_usec) / 1000000)+((float)(tv2.tv_sec - tv1.tv_sec));
        return executionSeconds;
    }
    
    else
    {
        return 0;
    }
}

