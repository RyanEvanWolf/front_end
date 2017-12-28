#include <front_end/WindowMatcher.hpp>

WindowMatcher::WindowMatcher(int windowsize)
{
	nWindow=windowsize;
	stereoSub=n.subscribe("front_end/stereo",10,&WindowMatcher::newStereo,this);	
	normSub=n.subscribe("front_end/normType",10,&WindowMatcher::updateNorm,this);
	encodingSub=n.subscribe("front_end/descriptor_encoding",10,&WindowMatcher::updateEncoding,this);
	windowPub=n.advertise<std_msgs::String>("front_end_window/window",20);

	it= new image_transport::ImageTransport(n);
	maskPub=it->advertise("front_end_window/leftMask",5);

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

	searchRegion=cv::Rect(0,0,100,100);

} 

void WindowMatcher::updateNorm(const std_msgs::Int8::ConstPtr& msg)
{
	normType.data=msg->data;
}

void WindowMatcher::updateEncoding(const std_msgs::String::ConstPtr& msg)
{
	encodingType=msg->data;
}


void WindowMatcher::newStereo(const front_end::StereoFrame::ConstPtr& msg)
{
	if(windowData.size()>=1)
	{
			int previousMatchesSize,currentMatchesSize;
			previousMatchesSize=windowData.back().size();
			currentMatchesSize=msg->matches.size();
			//build distance table left image
			cv::Mat leftMaskTable=cv::Mat(currentMatchesSize,previousMatchesSize,CV_8U);
			cv::Mat rightMaskTable=cv::Mat(currentMatchesSize,previousMatchesSize,CV_8U);
			for(int currentIndex=0;currentIndex<currentMatchesSize;currentIndex++)
			{
				for(int previousIndex=0;previousIndex<previousMatchesSize;previousIndex++)
				{
					//check within horizontal box, left images
					float leftx,lefty,rightx,righty;
					float currentlx,currently,currentrx,currentry;
					//compensate for ROI offset
					currentlx=msg->matches.at(currentIndex).leftFeature.imageCoord.x + lroi.x;
					currently=msg->matches.at(currentIndex).leftFeature.imageCoord.y + lroi.y;
					currentrx=msg->matches.at(currentIndex).rightFeature.imageCoord.x + rroi.x;
					currentry=msg->matches.at(currentIndex).rightFeature.imageCoord.y + rroi.y;
					
					leftx=windowData.back().at(previousIndex).leftFeature.imageCoord.x + lroi.x;
					lefty=windowData.back().at(previousIndex).leftFeature.imageCoord.y + lroi.y;
					rightx=windowData.back().at(previousIndex).rightFeature.imageCoord.x + rroi.x;
					righty=windowData.back().at(previousIndex).rightFeature.imageCoord.y + rroi.y;
					//check bounding box on the left image pair

					if((abs(currentlx-leftx)<searchRegion.width/2)&&(abs(currently-lefty)<searchRegion.height/2))
					{
						leftMaskTable.at<char>(currentIndex,previousIndex)=1;
					}
					else
					{
						leftMaskTable.at<char>(currentIndex,previousIndex)=0;
					}

					if((abs(currentrx-rightx)<searchRegion.width/2)&&(abs(currentry-righty)<searchRegion.height/2))
					{
						rightMaskTable.at<char>(currentIndex,previousIndex)=1;
					}
					else
					{
						rightMaskTable.at<char>(currentIndex,previousIndex)=0;
					}

				}
			}
			//rebuild left and image descriptors
			cv::Mat leftPrevDescr,leftCurrentDescr;
			cv::Mat rightPrevDescr,rightCurrentDescr;

			
			for(int row=0;row<currentMatchesSize;row++)
			{
				cv::Mat leftD,rightD;
				(cv_bridge::toCvCopy((msg->matches.at(row).leftFeature.descriptor),encodingType)->image).copyTo(leftD);
				(cv_bridge::toCvCopy((msg->matches.at(row).rightFeature.descriptor),encodingType)->image).copyTo(rightD);

				leftCurrentDescr.push_back(leftD);
				rightCurrentDescr.push_back(rightD);
			}
			for(int row=0;row<previousMatchesSize;row++)
			{
				cv::Mat leftD,rightD;
				(cv_bridge::toCvCopy((windowData.back().at(row).leftFeature.descriptor),encodingType)->image).copyTo(leftD);
				(cv_bridge::toCvCopy((windowData.back().at(row).rightFeature.descriptor),encodingType)->image).copyTo(rightD);

				leftPrevDescr.push_back(leftD);
				rightPrevDescr.push_back(rightD);
			}
			cv::BFMatcher m(normType.data,false);
			std::vector< std::vector<cv::DMatch> > initialLeftMatches,initialRightMatches;		
			m.knnMatch(leftCurrentDescr,leftPrevDescr,initialLeftMatches,2,leftMaskTable);
			m.knnMatch(rightCurrentDescr,rightPrevDescr,initialRightMatches,2,rightMaskTable);
			//lowe ratio
		/*	int ID=0;
		for(int index=0;index<initialMatches.size();index++)
		{
			if(initialMatches.at(index).size()>=2)
			{
				if(initialMatches.at(index).at(0).distance<0.8*initialMatches.at(index).at(1).distance)
				{
					front_end::StereoMatch current;
					cv::Mat ld,rd;//descriptor buffer
					cv::KeyPoint lkp,rkp;//keypoint buffer
					lkp=currentLeft.at(initialMatches.at(index).at(0).queryIdx);
					currentLeftDesc.row(initialMatches.at(index).at(0).queryIdx).copyTo(ld);
					current.leftFeature.imageCoord.x=lkp.pt.x;
					current.leftFeature.imageCoord.y=lkp.pt.y;

					cv_bridge::CvImage leftDescriptConversion(std_msgs::Header(),descriptorEncoding,ld);
					leftDescriptConversion.toImageMsg(current.leftFeature.descriptor);

					rkp=currentRight.at(initialMatches.at(index).at(0).trainIdx);
					currentRightDesc.row(initialMatches.at(index).at(0).trainIdx).copyTo(rd);
					current.rightFeature.imageCoord.x=rkp.pt.x;
					current.rightFeature.imageCoord.y=rkp.pt.y;

					cv_bridge::CvImage rightDescriptConversion(std_msgs::Header(),descriptorEncoding,rd);
					rightDescriptConversion.toImageMsg(current.rightFeature.descriptor);
	
					current.ID.data=ID;
					current.distance.data=initialMatches.at(index).at(0).distance;
					ID++;
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
					current.leftFeature.imageCoord.x=lkp.pt.x;
					current.leftFeature.imageCoord.y=lkp.pt.y;

					cv_bridge::CvImage leftDescriptConversion(std_msgs::Header(),descriptorEncoding,ld);
					leftDescriptConversion.toImageMsg(current.leftFeature.descriptor);

					rkp=currentRight.at(initialMatches.at(index).at(0).trainIdx);
					currentRightDesc.row(initialMatches.at(index).at(0).trainIdx).copyTo(rd);
					current.rightFeature.imageCoord.x=rkp.pt.x;
					current.rightFeature.imageCoord.y=rkp.pt.y;

					cv_bridge::CvImage rightDescriptConversion(std_msgs::Header(),descriptorEncoding,rd);
					rightDescriptConversion.toImageMsg(current.rightFeature.descriptor);
	
					current.ID.data=ID;
					current.distance.data=initialMatches.at(index).at(0).distance;
					ID++;
					outMessage.matches.push_back(current);
				}
			}
		}
			
*/
			//epiPolar rejection

			//publish Matches



			sensor_msgs::ImagePtr messageL=cv_bridge::CvImage(std_msgs::Header(),"8UC1",leftMaskTable).toImageMsg();
			maskPub.publish(messageL);
	}
	
	if(windowData.size()+1>nWindow)
	{
		windowData.pop_front();
	}
	windowData.push_back(msg->matches);
	std::cout<<windowData.size()<<std::endl;
	std_msgs::String output;
	output.data="yes";
	windowPub.publish(output);
}

