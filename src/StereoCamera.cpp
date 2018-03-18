#include "front_end/StereoCamera.hpp"
namespace stereo
{

StereoCamera::StereoCamera()
{
	stereoPub=n.advertise<front_end::StereoFrame>("front_end/stereo",20);
	normPub=n.advertise<std_msgs::Int8>("front_end/normType",20,true);
	encodingPub=n.advertise<std_msgs::String>("front_end/descriptor_encoding",20,true);
	offset_client=n.serviceClient<bumblebee::getOffset>("/bumblebee_configuration/getOffset");
	bumblebee::getOffset cameraoffset;
	offset_client.call(cameraoffset);
	lroi=cv::Rect(cameraoffset.response.left.x_offset,
								cameraoffset.response.left.y_offset,
								cameraoffset.response.left.width,
								cameraoffset.response.left.height);

	rroi=cv::Rect(cameraoffset.response.right.x_offset,
								cameraoffset.response.right.y_offset,
								cameraoffset.response.right.width,
								cameraoffset.response.right.height);
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
	
	while(ros::ok())
	{
		cv::Mat imageBuffer;
		boost::mutex::scoped_lock lock(mutexLImg);
    while(leftImages.empty())
    {
			leftImagesEmpty.wait(lock);
    }
		leftImages.front().copyTo(imageBuffer);
		leftImages.pop();
		lock.unlock();
		cv::Mat outDesc;
		std::vector<cv::KeyPoint> out;

		boost::mutex::scoped_lock lockDet(mutexlDet);
		
		lDet->detect(imageBuffer,out);
		lockDet.unlock();

		boost::mutex::scoped_lock lockDesc(mutexlDesc);
		lDesc->compute(imageBuffer,out,outDesc);
		lockDesc.unlock();

		//push features 
		boost::mutex::scoped_lock lockFeat(mutexLfeat);
		bool const was_empty=leftFeatures.empty();
  	leftFeatures.push(out);
		leftDescriptors.push(outDesc.clone());
  	if(was_empty)
		{
			leftFeaturesEmpty.notify_one();
		}
		out.clear();
		
	}
}

void StereoCamera::processRightImage()
{

	while(ros::ok())
	{
		cv::Mat imageBuffer;
		boost::mutex::scoped_lock lock(mutexRImg);
    while(rightImages.empty())
    {
			rightImagesEmpty.wait(lock);
    }
		rightImages.front().copyTo(imageBuffer);
		rightImages.pop();
		lock.unlock();
		cv::Mat outDesc;
		std::vector<cv::KeyPoint> out;

		boost::mutex::scoped_lock lockDet(mutexrDet);
		rDet->detect(imageBuffer,out);
		lockDet.unlock();

		boost::mutex::scoped_lock lockDesc(mutexrDesc);
		rDesc->compute(imageBuffer,out,outDesc);
		lockDesc.unlock();
		//push features 
		boost::mutex::scoped_lock lockFeat(mutexRfeat);
		bool const was_empty=rightFeatures.empty();
  	rightFeatures.push(out);
		rightDescriptors.push(outDesc.clone());
  	if(was_empty)
		{
			rightFeaturesEmpty.notify_one();
		}	
	}
}


void StereoCamera::processStereo()
{

	int frames=0;
	while(ros::ok())
	{
		std::vector<cv::KeyPoint> currentLeft,currentRight;
		cv::Mat currentLeftDesc,currentRightDesc;
		front_end::StereoFrame outMessage;
		//get left features from queue
		//wait for both queues to have atleast one set of features before processing
		boost::mutex::scoped_lock lockLf(mutexLfeat);
		while(leftFeatures.empty())
		{
			leftFeaturesEmpty.wait(lockLf);
		}
		currentLeft=leftFeatures.front();
		leftDescriptors.front().copyTo(currentLeftDesc);
		leftFeatures.pop();
		leftDescriptors.pop();
		lockLf.unlock();
		//get right features from queue
		boost::mutex::scoped_lock lockRf(mutexRfeat);
		while(rightFeatures.empty())
		{
			rightFeaturesEmpty.wait(lockRf);
		}
		currentRight=rightFeatures.front();
		rightDescriptors.front().copyTo(currentRightDesc);
		rightFeatures.pop();
		rightDescriptors.pop();
		lockRf.unlock();		

		//epipolar filter the points
		//build epipolar distance matrix
		outMessage.nLeft.data=currentLeft.size();
		outMessage.nRight.data=currentRight.size();
		

		cv::Mat maskTable=cv::Mat(currentLeft.size(),currentRight.size(),CV_8U);
		for(int leftIndex=0;leftIndex<currentLeft.size();leftIndex++)
		{
			for(int rightIndex=0;rightIndex<currentRight.size();rightIndex++)
			{
				if(abs(2*((currentLeft.at(leftIndex).pt.y+lroi.y)-(currentRight.at(rightIndex).pt.y+rroi.y)))<=2.0)
				{
					maskTable.at<uchar>(leftIndex,rightIndex)=1;
				}
				else
				{
					maskTable.at<uchar>(leftIndex,rightIndex)=0;
				}
			}
		}
		
		//match with mask as filter
		cv::BFMatcher m(normType.data,false);
		std::vector< std::vector<cv::DMatch> > initialMatches;		
		m.knnMatch(currentLeftDesc,currentRightDesc,initialMatches,2,maskTable);
		//only retain the two closest matches
		//filter with lowe ratio
		outMessage.loweRatio.data=initialMatches.size();

		std::vector<cv::DMatch>inlierMatches;

		for(int index=0;index<initialMatches.size();index++)
		{
			if(initialMatches.at(index).size()>=2)		
			{
				if(initialMatches.at(index).at(0).distance<0.8*initialMatches.at(index).at(1).distance)
				{
					bool found=false;
					//check it is unique in inlierMatches
					int inlierIndex=0;
					while(inlierIndex<inlierMatches.size()&&(!found))
					{
						if(initialMatches.at(index).at(0).queryIdx==inlierMatches.at(inlierIndex).queryIdx)
						{
							found=true;
							//find the lowest score
							if(initialMatches.at(index).at(0).distance<inlierMatches.at(inlierIndex).distance)
							{
								//swop them
								inlierMatches.at(inlierIndex)=initialMatches.at(index).at(0);
							}
						}
						inlierIndex++;
					}
					if(!found)
					{
						inlierMatches.push_back(initialMatches.at(index).at(0)); 
					}
				}
			}
			else
			{
				if(initialMatches.at(index).size()==1)
				{
					bool found=false;
					//check it is unique in inlierMatches
					int inlierIndex=0;
					while(inlierIndex<inlierMatches.size()&&(!found))
					{
						if(initialMatches.at(index).at(0).queryIdx==inlierMatches.at(inlierIndex).queryIdx)
						{
							found=true;
							//find the lowest score
							if(initialMatches.at(index).at(0).distance<inlierMatches.at(inlierIndex).distance)
							{
								//swop them
								inlierMatches.at(inlierIndex)=initialMatches.at(index).at(0);
							}
						}
						inlierIndex++;
					}
					if(!found)
					{
						inlierMatches.push_back(initialMatches.at(index).at(0)); 
					}
				}
			}
		}
		std::cout<<"here\n";
		for(int index=0;index<inlierMatches.size();index++)
		{
			front_end::StereoMatch current;
			cv::Mat ld,rd;//descriptor buffer
			cv::KeyPoint lkp,rkp;//keypoint buffer
			lkp=currentLeft.at(inlierMatches.at(index).queryIdx);
			currentLeftDesc.row(inlierMatches.at(index).queryIdx).copyTo(ld);
			current.leftFeature.imageCoord.x=lkp.pt.x +lroi.x;
			current.leftFeature.imageCoord.y=lkp.pt.y +lroi.y;

			cv_bridge::CvImage leftDescriptConversion(std_msgs::Header(),descriptorEncoding,ld);
			leftDescriptConversion.toImageMsg(current.leftFeature.descriptor);

			rkp=currentRight.at(inlierMatches.at(index).trainIdx);
			currentRightDesc.row(inlierMatches.at(index).trainIdx).copyTo(rd);
			current.rightFeature.imageCoord.x=rkp.pt.x+rroi.x;
			current.rightFeature.imageCoord.y=rkp.pt.y+rroi.y;

			cv_bridge::CvImage rightDescriptConversion(std_msgs::Header(),descriptorEncoding,rd);
			rightDescriptConversion.toImageMsg(current.rightFeature.descriptor);
	
			current.distance.data=inlierMatches.at(index).distance;
			outMessage.matches.push_back(current);
			
		}





	/*	for(int index=0;index<initialMatches.size();index++)
		{
			if(initialMatches.at(index).size()>=2)
			{
				if(initialMatches.at(index).at(0).distance<0.8*initialMatches.at(index).at(1).distance)
				{
					bool found=false;
					int inlierIndex=0;
					while(inlierIndex<inlierMatches.size()&&(!found))
					{
						if(initialmatches.at(index).at(0).queryIdx==inlierMatches.at(inlierIndex).queryIdx)
						{
							found=true;
							//find the lowest score
							if(initialmatches.at(index).at(0).distance<inlierMatches.at(inlierIndex).distance)
							{
								//swop them
								inlierMatches.at(inlierIndex)=initialmatches.at(index).at(0);
							}
						}
						inlierIndex++;
					}
					if(!found)
					{
						inlierMatches.push_back(initialmatches.at(index).at(0)); 
					}
			
					front_end::StereoMatch current;
					cv::Mat ld,rd;//descriptor buffer
					cv::KeyPoint lkp,rkp;//keypoint buffer
					lkp=currentLeft.at(initialMatches.at(index).at(0).queryIdx);
					currentLeftDesc.row(initialMatches.at(index).at(0).queryIdx).copyTo(ld);
					current.leftFeature.imageCoord.x=lkp.pt.x +lroi.x;
					current.leftFeature.imageCoord.y=lkp.pt.y +lroi.y;

					cv_bridge::CvImage leftDescriptConversion(std_msgs::Header(),descriptorEncoding,ld);
					leftDescriptConversion.toImageMsg(current.leftFeature.descriptor);

					rkp=currentRight.at(initialMatches.at(index).at(0).trainIdx);
					currentRightDesc.row(initialMatches.at(index).at(0).trainIdx).copyTo(rd);
					current.rightFeature.imageCoord.x=rkp.pt.x+rroi.x;
					current.rightFeature.imageCoord.y=rkp.pt.y+rroi.y;

					cv_bridge::CvImage rightDescriptConversion(std_msgs::Header(),descriptorEncoding,rd);
					rightDescriptConversion.toImageMsg(current.rightFeature.descriptor);
	
					current.distance.data=initialMatches.at(index).at(0).distance;
					outMessage.matches.push_back(current);
				}
			}
			else
			{
				if(initialMatches.at(index).size()==1)
				{
					front_end::StereoMatch current;
					cv::Mat ld,rd;//descriptor buffer
					cv::KeyPoint lkp,rkp;//keypoint buffer
					lkp=currentLeft.at(initialMatches.at(index).at(0).queryIdx);
					currentLeftDesc.row(initialMatches.at(index).at(0).queryIdx).copyTo(ld);
					current.leftFeature.imageCoord.x=lkp.pt.x +lroi.x;
					current.leftFeature.imageCoord.y=lkp.pt.y +lroi.y;

					cv_bridge::CvImage leftDescriptConversion(std_msgs::Header(),descriptorEncoding,ld);
					leftDescriptConversion.toImageMsg(current.leftFeature.descriptor);

					rkp=currentRight.at(initialMatches.at(index).at(0).trainIdx);
					currentRightDesc.row(initialMatches.at(index).at(0).trainIdx).copyTo(rd);
					current.rightFeature.imageCoord.x=rkp.pt.x+rroi.x;
					current.rightFeature.imageCoord.y=rkp.pt.y+rroi.y;

					cv_bridge::CvImage rightDescriptConversion(std_msgs::Header(),descriptorEncoding,rd);
					rightDescriptConversion.toImageMsg(current.rightFeature.descriptor);
					current.distance.data=initialMatches.at(index).at(0).distance;
					outMessage.matches.push_back(current);
				}
			}
		}*/
		std::cout<<outMessage.nLeft.data<<std::endl;
		std::cout<<outMessage.nRight.data<<std::endl;
		std::cout<<outMessage.loweRatio.data<<std::endl;
		std::cout<<outMessage.matches.size()<<std::endl;
		frames++;
		stereoPub.publish(outMessage);
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
	std::cout<<"set"<<std::endl;
	if(req.detection)
	{
		//set the feature detector
		if(name=="ORB")	
		{
			boost::mutex::scoped_lock lockL(mutexlDet);
			if(lDet.empty())
			{
				lDet=cv::FeatureDetector::create("ORB");
			}
			
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
			if(rDet.empty())
			{
				rDet=cv::FeatureDetector::create("ORB");
			}

			

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
		//set the descriptor extractor
		if(name=="ORB")
		{
			boost::mutex::scoped_lock lockL(mutexlDesc);
			if(lDesc.empty())
			{
				lDesc=cv::DescriptorExtractor::create("ORB");
			}
			
			lDesc->set("WTA_K",static_cast<int>(req.orbConfig.wta.data));
			lDesc->set("nFeatures",static_cast<int>(req.orbConfig.maxFeatures.data));
			lDesc->set("edgeThreshold",static_cast<int>(req.orbConfig.edge.data));
			lDesc->set("firstLevel",0);
			lDesc->set("nLevels",static_cast<int>(req.orbConfig.level.data));
			lDesc->set("patchSize",static_cast<int>(req.orbConfig.patch.data));
			lDesc->set("scaleFactor",static_cast<float>(req.orbConfig.scale.data));
			lDesc->set("scoreType",static_cast<int>(req.orbConfig.score.data));
			lockL.unlock();

			boost::mutex::scoped_lock lockR(mutexrDesc);
			if(rDesc.empty())
			{
				rDesc=cv::DescriptorExtractor::create("ORB");
			}
			
			rDesc->set("WTA_K",static_cast<int>(req.orbConfig.wta.data));
			rDesc->set("nFeatures",static_cast<int>(req.orbConfig.maxFeatures.data));
			rDesc->set("edgeThreshold",static_cast<int>(req.orbConfig.edge.data));
			rDesc->set("firstLevel",0);
			rDesc->set("nLevels",static_cast<int>(req.orbConfig.level.data));
			rDesc->set("patchSize",static_cast<int>(req.orbConfig.patch.data));
			rDesc->set("scaleFactor",static_cast<float>(req.orbConfig.scale.data));
			rDesc->set("scoreType",static_cast<int>(req.orbConfig.score.data));
			lockR.unlock();

			descriptorEncoding="8UC1";
			if(static_cast<int>(req.orbConfig.wta.data)>2)
			{
				normType.data=cv::NORM_HAMMING2;
			}
			else
			{	
				normType.data=cv::NORM_HAMMING;
			}

			std_msgs::String msg;
			msg.data=descriptorEncoding;
			encodingPub.publish(msg);
			normPub.publish(normType);
		}

	}
	return true;
}
	
}
