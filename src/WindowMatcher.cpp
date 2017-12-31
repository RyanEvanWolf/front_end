#include <front_end/WindowMatcher.hpp>

WindowMatcher::WindowMatcher(int windowsize)
{
	nWindow=windowsize;
	stereoSub=n.subscribe("front_end/stereo",10,&WindowMatcher::newStereo,this);	
	normSub=n.subscribe("front_end/normType",10,&WindowMatcher::updateNorm,this);
	encodingSub=n.subscribe("front_end/descriptor_encoding",10,&WindowMatcher::updateEncoding,this);
	windowPub=n.advertise<front_end::FrameTracks>("front_end_window/FrameTracks",20);
	statePub=n.advertise<front_end::Window>("front_end_window/State",20);

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

front_end::FrameTracks WindowMatcher::convertToMessage(std::vector<front_end::StereoMatch> currentMatches,
																													std::vector<front_end::StereoMatch> previousMatches,	
																													std::vector<cv::DMatch> matches)
{
	front_end::FrameTracks output;
	/*for(int matchesIndex=0;matchesIndex<matches.size();matchesIndex++)
	{
		std_msgs::Int32 prvInd,crrInd;
		crrInd.data=matches.at(matchesIndex).queryIdx;
		prvInd.data=matches.at(matchesIndex).trainIdx;
		output.previousFrameIndexes.push_back(prvInd);
		output.currentFrameIndexes.push_back(crrInd);	
	}*/
	return output;
}

front_end::FrameTracks WindowMatcher::extractMotion(std::vector<front_end::StereoMatch> currentMatches,
																													std::vector<front_end::StereoMatch> previousMatches,	
																													std::vector<cv::DMatch> matches)
{
		front_end::FrameTracks output;
	/*	cv::Mat currentPts,previousPts;
		//organize into cv::Mat format
		for(int index=0;index<matches.size();index++)
		{
				int currentIndex=matches.at(index).queryIdx;
				int previousIndex=matches.at(index).trainIdx;
				cv::Mat previous=cv::Mat(1,2,CV_64F);
				cv::Mat current=cv::Mat(1,2,CV_64F);
						
				current.at<double>(0,0)=currentMatches.at(currentIndex).leftFeature.imageCoord.x;
				current.at<double>(0,1)=currentMatches.at(currentIndex).leftFeature.imageCoord.y;
				currentPts.push_back(current);
				previous.at<double>(0,0)=previousMatches.at(previousIndex).leftFeature.imageCoord.x;
				previous.at<double>(0,1)=previousMatches.at(previousIndex).leftFeature.imageCoord.y;
				previousPts.push_back(previous);
		}
		//establish motion
		//organize into required format
		cv::Mat motionInlierMask;
		cv::Mat outR,outT,E;
		cv::Point2d pp(540.168,415.058);
		float fx=646.844;
		E = findEssentialMat(currentPts, previousPts, fx, pp, CV_RANSAC, 0.9, 0.1, motionInlierMask ); 
		recoverPose(E,currentPts,previousPts,outR,outT,fx,pp,motionInlierMask);
		std::vector<cv::DMatch> motionInliers;
		for(int index=0;index<matches.size();index++)
		{
			if(motionInlierMask.at<bool>(0,index))
			{
				motionInliers.push_back(matches.at(index));
			}
		}
		if(rotation.size()+1>(nWindow-1))
		{
			rotation.pop_front();
		}
		rotation.push_back(outR);
		if(translation.size()+1>(nWindow-1))
		{
			translation.pop_front();
		}
		translation.push_back(outT);*/


/*
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
		
		return output;//convertToMessage(currentMatches,previousMatches,motionInliers);
}

void WindowMatcher::publishCurrentState()
{
		front_end::Window message;
		/*std::vector<cv::Mat> triangulateMask;
		//add motion messages
		std::list<cv::Mat>::iterator it;
		for (it =  rotation.begin(); it !=  rotation.end(); ++it){
			cv_bridge::CvImage R(std_msgs::Header(),"64FC1",(*it));
			sensor_msgs::Image Rmess;
			R.toImageMsg(Rmess);
			message.rotation.push_back(Rmess);
		}
		for (it =  translation.begin(); it !=  translation.end(); ++it){
			cv_bridge::CvImage T(std_msgs::Header(),"64FC1",(*it));
			sensor_msgs::Image Tmess;
			T.toImageMsg(Tmess);
			message.translation.push_back(Tmess);
		}
		//add interframe tracks

		std::list<front_end::FrameTracks>::iterator itTrack;
		for(itTrack=initialData.begin();itTrack!=initialData.end();++itTrack)
		{
			message.lowe.push_back((*itTrack));
		}

		for(itTrack=motionData.begin();itTrack!=motionData.end();++itTrack)
		{
			message.inliers.push_back((*itTrack));
		}
		//calculate size of masks
		std::list<std::vector<front_end::StereoMatch>>::iterator itFrame;
		for(itFrame=windowData.begin();itFrame!=windowData.end();++itFrame)
		{
			for(int index=0;index<(*itFrame).size();index++)
			{
				message.window.push_back((*itFrame).at(index));
			}
		//	message.window.push_back((*itFrame));
			//cv::Mat frameMask=cv::Mat::zeros(0,(*itFrame).size(),CV_8U);
			//triangulateMask.push_back(frameMask.clone());
		}
*/
		statePub.publish(message);
}


void WindowMatcher::newStereo(const front_end::StereoFrame::ConstPtr& msg)
{
	front_end::FrameTracks output;
	if(windowData.size()>=1)
	{
		/*	int previousMatchesSize,currentMatchesSize;
			previousMatchesSize=windowData.back().size();
			currentMatchesSize=msg->matches.size();
			//build distance table left image
			cv::Mat leftMaskTable=getSearchMask(msg->matches,windowData.back());
			std::vector< std::vector<cv::DMatch> > initialLeftMatches=knnWindowMatch(msg->matches,		
																																							 windowData.back(),
																																							 leftMaskTable);
			//lowe ratio
			leftMaskTable=cv::Mat::zeros(currentMatchesSize,previousMatchesSize,CV_8U);
			std::vector<cv::DMatch> leftInliers=loweRejection(initialLeftMatches);		

			front_end::FrameTracks initialTracks;
			initialTracks=convertToMessage(msg->matches,
															windowData.back(),
															leftInliers);	
			//leftTracks.publish(initialTracks);
			output=extractMotion(msg->matches,
													windowData.back(),
													leftInliers);*/
		/*	cv::Mat previousPts,currentPts;
			for(int index=0;index<initialLeftMatches.size();index++)
			{

				if(initialLeftMatches.at(index).size()>=2)		
				{
					if(initialLeftMatches.at(index).at(0).distance<0.8*initialLeftMatches.at(index).at(1).distance)
					{
						//inlier
						int currentIndex=initialLeftMatches.at(index).at(0).queryIdx;
						int previousIndex=initialLeftMatches.at(index).at(0).trainIdx;
						leftMaskTable.at<char>(currentIndex,previousIndex)=1;
						cv::Mat previous=cv::Mat(1,2,CV_64F);
						cv::Mat current=cv::Mat(1,2,CV_64F);
						
						current.at<double>(0,0)=msg->matches.at(currentIndex).leftFeature.imageCoord.x;
						current.at<double>(0,1)=msg->matches.at(currentIndex).leftFeature.imageCoord.y;
						currentPts.push_back(current);

						previous.at<double>(0,0)=windowData.back().at(previousIndex).leftFeature.imageCoord.x;
						previous.at<double>(0,1)=windowData.back().at(previousIndex).leftFeature.imageCoord.y;
						previousPts.push_back(previous);
					}
				}
				else
				{
					if(initialLeftMatches.at(index).size()==1)
					{
						//inlier
						int currentIndex=initialLeftMatches.at(index).at(0).queryIdx;
						int previousIndex=initialLeftMatches.at(index).at(0).trainIdx;
						leftMaskTable.at<char>(currentIndex,previousIndex)=1;
						cv::Mat previous=cv::Mat(1,2,CV_64F);
						cv::Mat current=cv::Mat(1,2,CV_64F);
						
						current.at<double>(0,0)=msg->matches.at(currentIndex).leftFeature.imageCoord.x;
						current.at<double>(0,1)=msg->matches.at(currentIndex).leftFeature.imageCoord.y;
						currentPts.push_back(current);

						previous.at<double>(0,0)=windowData.back().at(previousIndex).leftFeature.imageCoord.x;
						previous.at<double>(0,1)=windowData.back().at(previousIndex).leftFeature.imageCoord.y;
						previousPts.push_back(previous);
					
					}
				}
			}

		//establish motion
		//organize into required format
		cv::Mat motionInlierMask;
		cv::Mat outR,outT,E;
		cv::Point2d pp(540.168,415.058);
		float fx=646.844;
		E = findEssentialMat(currentPts, previousPts, fx, pp, CV_RANSAC, 0.9, 5, motionInlierMask ); 
		recoverPose(E,currentPts,previousPts,outR,outT,fx,pp,motionInlierMask);

		int inlierIndex=0;
		for(int currentIndex=0;currentIndex<currentMatchesSize;currentIndex++)
		{
			for(int previousIndex=0;previousIndex<previousMatchesSize;previousIndex++)
			{
				if(leftMaskTable.at<bool>(currentIndex,previousIndex))
				{
					if(motionInlierMask.at<bool>(0,inlierIndex))
					{
						std_msgs::Int32 prvInd,crrInd;
						crrInd.data=currentIndex;
						prvInd.data=previousIndex;
						output.previousFrameIndexes.push_back(prvInd);
						output.currentFrameIndexes.push_back(crrInd);	
					}
					inlierIndex++;
				}
			}
		}

	//get the motion from the left Camera
*/
	/*


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


			//std::cout<<"inlierRatio = "<<float(output.previousFrameIndexes.size())/float(previousPts.rows)<<std::endl;
			//if(motionData.size()+1>(nWindow-1))
			//{
			//	motionData.pop_front();
		//	}
			//motionData.push_back(output);
			//if(initialData.size()+1>(nWindow-1))
			//{
			//	initialData.pop_front();
			//}
			//initialData.push_back(initialTracks);
	}
	//windowPub.publish(output);
	//if(windowData.size()+1>nWindow)
	//{
	//	windowData.pop_front();
	//}
	//windowData.push_back(msg->matches);
	publishCurrentState();

}



std::vector<cv::DMatch> WindowMatcher::loweRejection(std::vector< std::vector<cv::DMatch> > initial)
{
			std::vector<cv::DMatch>inlierMatches;

			for(int index=0;index<initial.size();index++)
			{
				if(initial.at(index).size()>=2)		
				{
					if(initial.at(index).at(0).distance<0.8*initial.at(index).at(1).distance)
					{
						bool found=false;
						//check it is unique in inlierMatches
						int inlierIndex=0;
						while(inlierIndex<inlierMatches.size()&&(!found))
						{
							if(initial.at(index).at(0).queryIdx==inlierMatches.at(inlierIndex).queryIdx)
							{
								found=true;
								//find the lowest score
								if(initial.at(index).at(0).distance<inlierMatches.at(inlierIndex).distance)
								{
									//swop them
									inlierMatches.at(inlierIndex)=initial.at(index).at(0);
								}
							}
							inlierIndex++;
						}
						if(!found)
						{
							inlierMatches.push_back(initial.at(index).at(0)); 
						}
					}
				}
				else
				{
					if(initial.at(index).size()==1)
					{
						bool found=false;
						//check it is unique in inlierMatches
						int inlierIndex=0;
						while(inlierIndex<inlierMatches.size()&&(!found))
						{
							if(initial.at(index).at(0).queryIdx==inlierMatches.at(inlierIndex).queryIdx)
							{
								found=true;
								//find the lowest score
								if(initial.at(index).at(0).distance<inlierMatches.at(inlierIndex).distance)
								{
									//swop them
									inlierMatches.at(inlierIndex)=initial.at(index).at(0);
								}
							}
							inlierIndex++;
						}
						if(!found)
						{
							inlierMatches.push_back(initial.at(index).at(0)); 
						}
					}
				}
			}
	return inlierMatches;
}
std::vector< std::vector<cv::DMatch> > WindowMatcher::knnWindowMatch(std::vector<front_end::StereoMatch> currentMatches,
																													std::vector<front_end::StereoMatch> previousMatches,	
																													cv::Mat mask)
{
		cv::Mat PrevDescr,CurrentDescr;
			
			for(int row=0;row<currentMatches.size();row++)
			{
				cv::Mat D;
				(cv_bridge::toCvCopy((currentMatches.at(row).leftFeature.descriptor),encodingType)->image).copyTo(D);
				CurrentDescr.push_back(D);
			}
			for(int row=0;row<previousMatches.size();row++)
			{
				cv::Mat D;
				(cv_bridge::toCvCopy((previousMatches.at(row).leftFeature.descriptor),encodingType)->image).copyTo(D);
				PrevDescr.push_back(D);
			}
			cv::BFMatcher m(normType.data,false);
			std::vector< std::vector<cv::DMatch> > matches;
			m.knnMatch(CurrentDescr,PrevDescr,matches,2,mask);
			return matches;
}




cv::Mat WindowMatcher::getSearchMask(std::vector<front_end::StereoMatch> currentMatches,std::vector<front_end::StereoMatch> previousMatches)
{
			//build distance table left image
			cv::Mat maskTable=cv::Mat(currentMatches.size(),previousMatches.size(),CV_8U);
			for(int currentIndex=0;currentIndex<currentMatches.size();currentIndex++)
			{
				for(int previousIndex=0;previousIndex<previousMatches.size();previousIndex++)
				{
					//check within horizontal box, left images
					float leftx,lefty;
					float currentlx,currently;
					//declaring extra variable purely for formatting purposes
					currentlx=currentMatches.at(currentIndex).leftFeature.imageCoord.x;
					currently=currentMatches.at(currentIndex).leftFeature.imageCoord.y;

					
					leftx=previousMatches.at(previousIndex).leftFeature.imageCoord.x;
					lefty=previousMatches.at(previousIndex).leftFeature.imageCoord.y;


					if((abs(currentlx-leftx)<searchRegion.width/2)&&(abs(currently-lefty)<searchRegion.height/2))
					{
						maskTable.at<char>(currentIndex,previousIndex)=1;
					}
					else
					{
						maskTable.at<char>(currentIndex,previousIndex)=0;
					}
				}
			}	
	return maskTable;
}
