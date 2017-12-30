#include <front_end/WindowMatcher.hpp>

WindowMatcher::WindowMatcher(int windowsize)
{
	nWindow=windowsize;
	stereoSub=n.subscribe("front_end/stereo",10,&WindowMatcher::newStereo,this);	
	normSub=n.subscribe("front_end/normType",10,&WindowMatcher::updateNorm,this);
	encodingSub=n.subscribe("front_end/descriptor_encoding",10,&WindowMatcher::updateEncoding,this);
	windowPub=n.advertise<front_end::FrameTracks>("front_end_window/FrameTracks",20);
	leftTracks=n.advertise<front_end::FrameTracks>("front_end_window/FrameTracksleft",20);
	rightTracks=n.advertise<front_end::FrameTracks>("front_end_window/FrameTracksright",20);


	it= new image_transport::ImageTransport(n);


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
	front_end::FrameTracks output;
	if(windowData.size()>=1)
	{
			front_end::FrameTracks ltrack,rtrack;
			int previousMatchesSize,currentMatchesSize;
			previousMatchesSize=windowData.back().size();
			currentMatchesSize=msg->matches.size();
			//build distance table left image
			cv::Mat leftMaskTable=cv::Mat(currentMatchesSize,previousMatchesSize,CV_8U);
			//cv::Mat rightMaskTable=cv::Mat(currentMatchesSize,previousMatchesSize,CV_8U);
			for(int currentIndex=0;currentIndex<currentMatchesSize;currentIndex++)
			{
				for(int previousIndex=0;previousIndex<previousMatchesSize;previousIndex++)
				{
					//check within horizontal box, left images
					float leftx,lefty,rightx,righty;
					float currentlx,currently,currentrx,currentry;
					//compensate for ROI offset
					currentlx=msg->matches.at(currentIndex).leftFeature.imageCoord.x;
					currently=msg->matches.at(currentIndex).leftFeature.imageCoord.y;
					//currentrx=msg->matches.at(currentIndex).rightFeature.imageCoord.x;
					//currentry=msg->matches.at(currentIndex).rightFeature.imageCoord.y;
					
					leftx=windowData.back().at(previousIndex).leftFeature.imageCoord.x;
					lefty=windowData.back().at(previousIndex).leftFeature.imageCoord.y;
					//rightx=windowData.back().at(previousIndex).rightFeature.imageCoord.x;
					//righty=windowData.back().at(previousIndex).rightFeature.imageCoord.y;
					//check bounding box on the left image pair

					if((abs(currentlx-leftx)<searchRegion.width/2)&&(abs(currently-lefty)<searchRegion.height/2))
					{
						leftMaskTable.at<char>(currentIndex,previousIndex)=1;
					}
					else
					{
						leftMaskTable.at<char>(currentIndex,previousIndex)=0;
					}

				/*	if((abs(currentrx-rightx)<searchRegion.width/2)&&(abs(currentry-righty)<searchRegion.height/2))
					{
						rightMaskTable.at<char>(currentIndex,previousIndex)=1;
					}
					else
					{
						rightMaskTable.at<char>(currentIndex,previousIndex)=0;
					}
*/
				}
			}
			//rebuild left and image descriptors
			cv::Mat leftPrevDescr,leftCurrentDescr;
		//	cv::Mat rightPrevDescr,rightCurrentDescr;

			
			for(int row=0;row<currentMatchesSize;row++)
			{
				cv::Mat leftD,rightD;
				(cv_bridge::toCvCopy((msg->matches.at(row).leftFeature.descriptor),encodingType)->image).copyTo(leftD);
				//(cv_bridge::toCvCopy((msg->matches.at(row).rightFeature.descriptor),encodingType)->image).copyTo(rightD);

				leftCurrentDescr.push_back(leftD);
				//rightCurrentDescr.push_back(rightD);
			}
			for(int row=0;row<previousMatchesSize;row++)
			{
				cv::Mat leftD,rightD;
				(cv_bridge::toCvCopy((windowData.back().at(row).leftFeature.descriptor),encodingType)->image).copyTo(leftD);
				//(cv_bridge::toCvCopy((windowData.back().at(row).rightFeature.descriptor),encodingType)->image).copyTo(rightD);

				leftPrevDescr.push_back(leftD);
				//rightPrevDescr.push_back(rightD);
			}
			cv::BFMatcher m(normType.data,false);
			std::vector< std::vector<cv::DMatch> > initialLeftMatches,initialRightMatches;
			std::vector<cv::DMatch> leftInliers,rightInliers;
			std::vector<cv::DMatch> combinedInliers;
			m.knnMatch(leftCurrentDescr,leftPrevDescr,initialLeftMatches,2,leftMaskTable);
			//m.knnMatch(rightCurrentDescr,rightPrevDescr,initialRightMatches,2,rightMaskTable);


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
		/*	leftTracks.publish(ltrack);
			for(int index=0;index<initialRightMatches.size();index++)
			{
				if(initialRightMatches.at(index).size()>=2)		
				{
					if(initialRightMatches.at(index).at(0).distance<0.8*initialRightMatches.at(index).at(1).distance)
					{
						//inlier
						rightInliers.push_back(initialRightMatches.at(index).at(0));
						std_msgs::Int32 prvInd,crrInd;
						crrInd.data=initialRightMatches.at(index).at(0).queryIdx;
						prvInd.data=initialRightMatches.at(index).at(0).trainIdx;

						rtrack.previousFrameIndexes.push_back(prvInd);
						rtrack.currentFrameIndexes.push_back(crrInd);
					}
				}
				else
				{
					if(initialRightMatches.at(index).size()==1)
					{
												//inlier
						rightInliers.push_back(initialRightMatches.at(index).at(0));
						std_msgs::Int32 prvInd,crrInd;
						crrInd.data=initialRightMatches.at(index).at(0).queryIdx;
						prvInd.data=initialRightMatches.at(index).at(0).trainIdx;

						rtrack.previousFrameIndexes.push_back(prvInd);
						rtrack.currentFrameIndexes.push_back(crrInd);
					}
				}
			}
			rightTracks.publish(rtrack);*/
			//correspondence rejection
		
		/* very buggy, something wrong with the indeces
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
			}*/
		//establish motion
		//organize into required format
		cv::Mat	previousInliers=cv::Mat(output.previousFrameIndexes.size(),2,CV_64F);
		cv::Mat	currentInliers=cv::Mat(output.currentFrameIndexes.size(),2,CV_64F);
	
		for(int index=0;index<output.currentFrameIndexes.size();index++)
		{
			currentInliers.at<double>(index,0)=req.current.left.features.at(index).x/req.current.left.features.at(index).z;
			currentInliers.at<double>(index,1)=req.current.left.features.at(index).y/req.current.left.features.at(index).z;
		}
		for(int index=0;index<output.previousFrameIndexes.size();index++)
		{
			previousInliers.at<double>(index,0)=req.previous.right.features.at(index).x/req.previous.right.features.at(index).z;
			previousInliers.at<double>(index,1)=req.previous.right.features.at(index).y/req.previous.right.features.at(index).z;
		}

	//get the motion from the left Camera

	/*cv::Mat prevPoints,currentPoints;
	prevPoints=cv::Mat(req.previous.left.features.size(),2,CV_64F);
	currentPoints=cv::Mat(req.current.right.features.size(),2,CV_64F);

	cv::Point2d pp(5, 8); 
	cv::Mat mask;


	cv::Mat outR,outT;
	cv::Mat E = findEssentialMat(currentPoints, prevPoints, 300.0, pp, CV_RANSAC, 0.99, 3, mask ); 
	recoverPose(E,currentPoints,prevPoints,outR,outT,300.0,pp,mask);

	cv::Mat P=cv::Mat::zeros(3,4,CV_64F);
	P.at<double>(0,0)=300.0;
	P.at<double>(0,2)=5;
	P.at<double>(1,1)=300.0;
	P.at<double>(1,2)=8;
	P.at<double>(2,2)=1;
	cv::Mat k=P(cv::Rect(0,0,3,3));


	int totalAverageSamples=0;

	cv::Mat average=cv::Mat::zeros(3,1,CV_64F);

	for(int index=0;index<req.current.left.landmarks.size();index++)
	{

		if(mask.at<bool>(0,index))
		{
			cv::Mat xnew,xold;
			//compute scale from projection 
			//projection pixel in previous frame
			xold=cv::Mat(3,1,CV_64F);
			xold.at<double>(0,0)=req.previous.right.features.at(index).x;
			xold.at<double>(1,0)=req.previous.right.features.at(index).y;
			xold.at<double>(2,0)=req.previous.right.features.at(index).z;

			xnew=cv::Mat(3,1,CV_64F);
			xnew.at<double>(0,0)=req.current.left.landmarks.at(index).x;
			xnew.at<double>(1,0)=req.current.left.landmarks.at(index).y;
			xnew.at<double>(2,0)=req.current.left.landmarks.at(index).z;
			average+=((k.inv()*xold-outR*xnew)*outT.inv(cv::DECOMP_SVD))*outT;
			totalAverageSamples++;
			if(totalAverageSamples==3)
			{
				index=req.current.left.landmarks.size();
			}
		}
	}

	if(totalAverageSamples>0)
	{
		average=average/double(totalAverageSamples); 
		//store and send back output in a ros message
		for(int row=0;row<3;row++)
		{
			for(int column=0;column<3;column++)
			{
				res.R.data[3*row +column]=outR.at<double>(row,column);
			}
		}
		res.T.x=average.at<double>(0,0);
		res.T.y=average.at<double>(1,0);
		res.T.z=average.at<double>(2,0);
		
		for(int index=0;index<mask.cols;index++)
		{
			res.mask.push_back(mask.at<bool>(0,index));	
		}
	}
	else
	{

		std::cout<<"no Inliers detected"<<std::endl;
	}*/


			std::cout<<"Ntracks = "<<output.previousFrameIndexes.size()<<std::endl;

	}
	windowPub.publish(output);
	if(windowData.size()+1>nWindow)
	{
		windowData.pop_front();
	}
	windowData.push_back(msg->matches);
}

