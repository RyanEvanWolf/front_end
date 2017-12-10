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
	boost::thread stereoThread(boost::bind(&StereoCamera::processStereo,this));
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
	cv::Mat imageBuffer;
	while(ros::ok())
	{
		boost::mutex::scoped_lock lock(mutexLImg);
    while(leftImages.empty())
    {
			leftImagesEmpty.wait(lock);
    }
		leftImages.front().copyTo(imageBuffer);
		leftImages.pop();
		lock.unlock();

		boost::mutex::scoped_lock lockDet(mutexlDet);
		std::vector<cv::KeyPoint> out;
		lDet->detect(imageBuffer,out);
		lockDet.unlock();
		//push features 
		boost::mutex::scoped_lock lockFeat(mutexLfeat);
		bool const was_empty=leftFeatures.empty();
  	leftFeatures.push(out);
  	if(was_empty)
		{
			leftFeaturesEmpty.notify_one();
		}
	}
}

void StereoCamera::processRightImage()
{
	cv::Mat imageBuffer;
	while(ros::ok())
	{
		boost::mutex::scoped_lock lock(mutexRImg);
    while(rightImages.empty())
    {
			rightImagesEmpty.wait(lock);
    }
		rightImages.front().copyTo(imageBuffer);
		rightImages.pop();
		std::vector<cv::KeyPoint> out;
		lock.unlock();

		boost::mutex::scoped_lock lockDet(mutexrDet);
		rDet->detect(imageBuffer,out);
		lockDet.unlock();
		//push features 
		boost::mutex::scoped_lock lockFeat(mutexRfeat);
		bool const was_empty=rightFeatures.empty();
  	rightFeatures.push(out);
  	if(was_empty)
		{
			rightFeaturesEmpty.notify_one();
		}		
	}
}

void StereoCamera::processStereo()
{
	std::vector<cv::KeyPoint> currentLeft,currentRight;
	int frames=0;
	while(ros::ok())
	{
		//get left features from queue
		//wait for both queues to have atleast one set of features before processing
		boost::mutex::scoped_lock lockLf(mutexLfeat);
		while(leftFeatures.empty())
		{
			leftFeaturesEmpty.wait(lockLf);
		}
		currentLeft=leftFeatures.front();
		leftFeatures.pop();
		lockLf.unlock();
		//get right features from queue
		boost::mutex::scoped_lock lockRf(mutexRfeat);
		while(rightFeatures.empty())
		{
			rightFeaturesEmpty.wait(lockRf);
		}
		currentRight=rightFeatures.front();
		rightFeatures.pop();
		lockRf.unlock();		
		//epipolar filter the points
		frames++;
		std::cout<<"frames Processed -> "<<frames<<std::endl;
	}
}

/*void printParams( cv::Algorithm* algorithm ) {
    std::vector<std::string> parameters;
    algorithm->getParams(parameters);

    for (int i = 0; i < (int) parameters.size(); i++) {
        std::string param = parameters[i];
        int type = algorithm->paramType(param);
        std::string helpText = algorithm->paramHelp(param);
        std::string typeText;

        switch (type) {
        case cv::Param::BOOLEAN:
            typeText = "bool";
            break;
        case cv::Param::INT:
            typeText = "int";
            break;
        case cv::Param::REAL:
            typeText = "real (double)";
            break;
        case cv::Param::STRING:
            typeText = "string";
            break;
        case cv::Param::MAT:
            typeText = "Mat";
            break;
        case cv::Param::ALGORITHM:
            typeText = "Algorithm";
            break;
        case cv::Param::MAT_VECTOR:
            typeText = "Mat vector";
            break;
        }
        std::cout << "Parameter '" << param << "' type=" << typeText << " help=" << helpText << std::endl;
    }
}
*/


bool StereoCamera::updateDetector(front_end::setDetector::Request& req,front_end::setDetector::Response &res)
{
	
	std::string name=(req.Name.data);
	
	if(req.detection)
	{
		if(name=="ORB")	
		{
			boost::mutex::scoped_lock lockL(mutexlDet);
			lDet=cv::FeatureDetector::create("ORB");
			lDet->set("WTA_K",static_cast<int>(req.orbConfig.wta.data));
			lDet->set("nFeatures",static_cast<int>(req.orbConfig.maxFeatures.data));
			lDet->set("edgeThreshold",static_cast<int>(req.orbConfig.edge.data));
			lDet->set("firstLevel",0);
			lDet->set("nLevels",static_cast<int>(req.orbConfig.level.data));
			lDet->set("patchSize",static_cast<int>(req.orbConfig.patch.data));
			lDet->set("scaleFactor",static_cast<float>(req.orbConfig.scale.data));
			lDet->set("scoreType",static_cast<int>(req.orbConfig.score.data));
			lockL.unlock();

			boost::mutex::scoped_lock lockR(mutexrDet);

			rDet=cv::FeatureDetector::create("ORB");
			rDet->set("WTA_K",static_cast<int>(req.orbConfig.wta.data));
			rDet->set("nFeatures",static_cast<int>(req.orbConfig.maxFeatures.data));
			rDet->set("edgeThreshold",static_cast<int>(req.orbConfig.edge.data));
			rDet->set("firstLevel",0);
			rDet->set("nLevels",static_cast<int>(req.orbConfig.level.data));
			rDet->set("patchSize",static_cast<int>(req.orbConfig.patch.data));
			rDet->set("scaleFactor",static_cast<float>(req.orbConfig.scale.data));
			rDet->set("scoreType",static_cast<int>(req.orbConfig.score.data));
			lockR.unlock();
		}
	}
	else
	{

	}
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
