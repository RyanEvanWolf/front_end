#include <front_end/WindowMatcher.hpp>

WindowMatcher::WindowMatcher(int windowsize)
{
	debug=0;
	nWindow=windowsize;
	stereoSub=n.subscribe("front_end/stereo",10,&WindowMatcher::newStereo,this);	
	normSub=n.subscribe("front_end/normType",10,&WindowMatcher::updateNorm,this);
	encodingSub=n.subscribe("front_end/descriptor_encoding",10,&WindowMatcher::updateEncoding,this);
	windowPub=n.advertise<front_end::FrameTracks>("front_end_window/FrameTracks",20);

	it= new image_transport::ImageTransport(n);
	//maskPub=it->advertise("front_end_window/leftMask",5);

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
			front_end::FrameTracks output;
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
			std::vector<cv::DMatch> leftInliers,rightInliers;
			std::vector<cv::DMatch> combinedInliers;
			m.knnMatch(leftCurrentDescr,leftPrevDescr,initialLeftMatches,2,leftMaskTable);
			m.knnMatch(rightCurrentDescr,rightPrevDescr,initialRightMatches,2,rightMaskTable);


			//lowe ratio
			for(int index=0;index<initialLeftMatches.size();index++)
			{
				if(initialLeftMatches.at(index).size()>=2)		
				{
					if(initialLeftMatches.at(index).at(0).distance<0.8*initialLeftMatches.at(index).at(1).distance)
					{
						//inlier
						leftInliers.push_back(initialLeftMatches.at(index).at(0));
						std_msgs::Int32 prvInd,crrInd;
						crrInd.data=initialLeftMatches.at(index).at(0).queryIdx;
						prvInd.data=initialLeftMatches.at(index).at(0).trainIdx;
					
						output.previousFrameIndexes.push_back(prvInd);
						output.currentFrameIndexes.push_back(crrInd);
					}
				}
				else
				{
					if(initialLeftMatches.at(index).size()==1)
					{
												//inlier
						leftInliers.push_back(initialLeftMatches.at(index).at(0));
						std_msgs::Int32 prvInd,crrInd;
						crrInd.data=initialLeftMatches.at(index).at(0).queryIdx;
						prvInd.data=initialLeftMatches.at(index).at(0).trainIdx;
					
						output.previousFrameIndexes.push_back(prvInd);
						output.currentFrameIndexes.push_back(crrInd);
					}
				}
			}

			for(int index=0;index<initialRightMatches.size();index++)
			{
				if(initialRightMatches.at(index).size()>=2)		
				{
					if(initialRightMatches.at(index).at(0).distance<0.8*initialRightMatches.at(index).at(1).distance)
					{
						//inlier
						rightInliers.push_back(initialRightMatches.at(index).at(0));
						//std_msgs::Int32 prvInd,crrInd;
						//crrInd.data=initialLeftMatches.at(index).at(0).queryIdx;
						//prvInd.data=initialLeftMatches.at(index).at(0).trainIdx;
					
						//output.previousFrameIndexes.push_back(prvInd);
						//output.currentFrameIndexes.push_back(crrInd);
					}
				}
				else
				{
					if(initialRightMatches.at(index).size()==1)
					{
												//inlier
						rightInliers.push_back(initialRightMatches.at(index).at(0));
						//std_msgs::Int32 prvInd,crrInd;
						//crrInd.data=initialLeftMatches.at(index).at(0).queryIdx;
						//prvInd.data=initialLeftMatches.at(index).at(0).trainIdx;
					
						//output.previousFrameIndexes.push_back(prvInd);
						//output.currentFrameIndexes.push_back(crrInd);
					}
				}
			}
			std::cout<<leftInliers.size()<<"\t"<<rightInliers.size()<<std::endl;
			//correspondence rejection
			for(int leftIndex=0;leftIndex<leftInliers.size();leftIndex++)
			{
				//check if both have same index found
				int previousStereoIndex=leftInliers.at(leftIndex).trainIdx;
				for(int rightIndex=0;rightIndex<rightInliers.size();rightIndex++)
				{	
					//std::cout<<bool(previousStereoIndex==rightInliers.at(rightIndex).trainIdx);
					if(previousStereoIndex==rightInliers.at(rightIndex).trainIdx)
					{
						std_msgs::Int32 prvInd,crrInd;
						crrInd.data=leftInliers.at(leftIndex).queryIdx;
						prvInd.data=leftInliers.at(leftIndex).trainIdx;

						output.previousFrameIndexes.push_back(prvInd);
						output.currentFrameIndexes.push_back(crrInd);
					}
				}
			}

			std::cout<<output.previousFrameIndexes.size()<<std::endl;
			windowPub.publish(output);

	}
	
	if(windowData.size()+1>nWindow)
	{
		windowData.pop_front();
	}
	windowData.push_back(msg->matches);
	std::cout<<debug<<std::endl;
	debug++;
}

