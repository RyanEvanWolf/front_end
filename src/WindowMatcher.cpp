#include <front_end/WindowMatcher.hpp>

WindowMatcher::WindowMatcher(int windowsize)
{
	debug=0;
	nWindow=windowsize;

	stereoSub=n.subscribe("front_end/stereo",10,&WindowMatcher::newStereo,this);	
	normSub=n.subscribe("front_end/normType",10,&WindowMatcher::updateNorm,this);
	encodingSub=n.subscribe("front_end/descriptor_encoding",10,&WindowMatcher::updateEncoding,this);
	getQmapClient=n.serviceClient<bumblebee::getQ>("/bumblebee_configuration/getQ");
	cameraSub=n.subscribe("/bumblebee_configuration/left/info",10,&WindowMatcher::newCamera,this);


	bumblebee::getQ message;
	getQmapClient.call(message);
	Q=cv::Mat::zeros(4,4,CV_64F);
	for(int row=0;row<4;row++)
	{
		for(int col=0;col<4;col++)
		{
			Q.at<double>(row,col)=message.response.Q[row*4+col];
		}
	}

	windowPub=n.advertise<front_end::FrameTracks>("front_end_window/FrameTracks",20);
	statePub=n.advertise<front_end::Window>("front_end_window/State",20);

	it= new image_transport::ImageTransport(n);


	searchRegion=cv::Rect(0,0,100,100);

}

void WindowMatcher::triangulate(front_end::Landmark &in)
{
	cv::Mat homogPoint,inputPoint;
	inputPoint=cv::Mat(4,1,CV_64F);
	
	inputPoint.at<double>(0,0)=in.stereo.leftFeature.imageCoord.x;
	inputPoint.at<double>(1,0)=in.stereo.leftFeature.imageCoord.y;
	inputPoint.at<double>(2,0)=in.stereo.leftFeature.imageCoord.x-in.stereo.rightFeature.imageCoord.x;
	inputPoint.at<double>(3,0)=1.0;

	homogPoint=Q*inputPoint;
		
	in.distance.x=homogPoint.at<double>(0,0)/(1000*homogPoint.at<double>(3,0));//in meters
	in.distance.y=homogPoint.at<double>(1,0)/(1000*homogPoint.at<double>(3,0));
	in.distance.z=homogPoint.at<double>(2,0)/(1000*homogPoint.at<double>(3,0));
} 

void WindowMatcher::updateNorm(const std_msgs::Int8::ConstPtr& msg)
{
	normType.data=msg->data;
}

void WindowMatcher::updateEncoding(const std_msgs::String::ConstPtr& msg)
{
	encodingType=msg->data;
}

void WindowMatcher::newCamera(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
		P=cv::Mat(3,4,CV_64F);
		for(int row=0;row<3;row++)
		{
			for(int col=0;col<4;col++)
			{
				P.at<double>(row,col)=msg->P[row*4+col];
			}
		}
}

void WindowMatcher::newStereo(const front_end::StereoFrame::ConstPtr& msg)
{
	front_end::WindowFrame current,previous;
	auto start=std::chrono::steady_clock::now();
	for(int index=0;index<msg->matches.size();index++)
	{
		front_end::Landmark buffer;
		buffer.stereo=msg->matches.at(index);
		triangulate(buffer);
		current.inliers.push_back(buffer);
	}
	auto end=std::chrono::steady_clock::now();
	double timeTaken=std::chrono::duration<double,std::milli>(end-start).count();
	current.timings.push_back(timeTaken);
	current.timingDescription.push_back("Triangulation");	


	window.push_back(current);
	if(window.size()>=nWindow)
	{
		window.erase(window.begin());
	}

	if(window.size()>1)
	{
		previous=window.at(window.size()-2);
		start=std::chrono::steady_clock::now();
		front_end::InterWindowFrame latestInter;
		//calculate potential Window matches mask
		cv::Mat leftMaskTable=cv::Mat(current.inliers.size(),previous.inliers.size(),CV_8U);
		for(int currentIndex=0;currentIndex<current.inliers.size();currentIndex++)
		{
			for(int previousIndex=0;previousIndex<previous.inliers.size();previousIndex++)
			{
				//check within horizontal box, left images
				float leftx,lefty;
				float currentlx,currently;
				//declaring extra variable purely for formatting purposes
				currentlx=current.inliers.at(currentIndex).stereo.leftFeature.imageCoord.x;
				currently=current.inliers.at(currentIndex).stereo.leftFeature.imageCoord.y;
					
				leftx=previous.inliers.at(previousIndex).stereo.leftFeature.imageCoord.x;
				lefty=previous.inliers.at(previousIndex).stereo.leftFeature.imageCoord.y;

				if((abs(currentlx-leftx)<searchRegion.width/2)&&(abs(currently-lefty)<searchRegion.height/2))
				{
					leftMaskTable.at<char>(currentIndex,previousIndex)=1;
				}
				else
				{
					leftMaskTable.at<char>(currentIndex,previousIndex)=0;
				}
			}
		}
		end=std::chrono::steady_clock::now();
		timeTaken=std::chrono::duration<double,std::milli>(end-start).count();
		current.timings.push_back(timeTaken);
		current.timingDescription.push_back("ROIsearch");	
		//Perform KNN matching with lowe rejection 
		cv::Mat PrevDescr,CurrentDescr;
		
		//not including the copying costs in the time measurement
		for(int row=0;row<current.inliers.size();row++)
		{
			cv::Mat D;
			(cv_bridge::toCvCopy((current.inliers.at(row).stereo.leftFeature.descriptor),encodingType)->image).copyTo(D);
			CurrentDescr.push_back(D);
		}
		for(int row=0;row<previous.inliers.size();row++)
		{
			cv::Mat D;
			(cv_bridge::toCvCopy((previous.inliers.at(row).stereo.leftFeature.descriptor),encodingType)->image).copyTo(D);
			PrevDescr.push_back(D);
		}
		
		cv::BFMatcher m(normType.data,false);
		start=std::chrono::steady_clock::now();
		std::vector< std::vector<cv::DMatch> > initialmatches;
		m.knnMatch(CurrentDescr,PrevDescr,initialmatches,2,leftMaskTable);
		end=std::chrono::steady_clock::now();
		timeTaken=std::chrono::duration<double,std::milli>(end-start).count();
		latestInter.timings.push_back(timeTaken);
		latestInter.timingDescription.push_back("KNN_Match");


		//perform lowe ratio	
		std::vector<cv::DMatch>inlierMatches;
		start=std::chrono::steady_clock::now();
		
		for(int index=0;index<initialmatches.size();index++)
		{
			if(initialmatches.at(index).size()>=2)		
			{
				if(initialmatches.at(index).at(0).distance<0.8*initialmatches.at(index).at(1).distance)
				{
					bool found=false;
					//check it is unique in inlierMatches
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
				}
			}
			else
			{
				if(initialmatches.at(index).size()==1)
				{
					bool found=false;
					//check it is unique in inlierMatches
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
				}
			}
		}
		end=std::chrono::steady_clock::now();
		timeTaken=std::chrono::duration<double,std::milli>(end-start).count();
		latestInter.timings.push_back(timeTaken);
		latestInter.timingDescription.push_back("LoweRatio");

		//populate message indexes
		for(int index=0;index<inlierMatches.size();index++)
		{
			latestInter.currentInlierIndexes.push_back(inlierMatches.at(index).queryIdx);
			latestInter.previousInlierIndexes.push_back(inlierMatches.at(index).trainIdx);
		}
		//calculate motion and motionInlier Mask
		//organize into the correct pts format 
		std::cout<<"getting Motion at frame = "<<debug<<std::endl;
		cv::Mat currentPts,previousPts;
		for(int index=0;index<latestInter.currentInlierIndexes.size();index++)
		{
				cv::Mat p=cv::Mat(1,2,CV_64F);
				cv::Mat c=cv::Mat(1,2,CV_64F);
				c.at<double>(0,0)=current.inliers.at(latestInter.currentInlierIndexes.at(index)).stereo.leftFeature.imageCoord.x;
				c.at<double>(0,1)=current.inliers.at(latestInter.currentInlierIndexes.at(index)).stereo.leftFeature.imageCoord.y;
				currentPts.push_back(c);
				p.at<double>(0,0)=previous.inliers.at(latestInter.previousInlierIndexes.at(index)).stereo.leftFeature.imageCoord.x;
				p.at<double>(0,1)=previous.inliers.at(latestInter.previousInlierIndexes.at(index)).stereo.leftFeature.imageCoord.y;
				previousPts.push_back(p);			
		}
		start=std::chrono::steady_clock::now();		
		cv::Mat motionInlierMask;
		cv::Mat outR,outT,E;
		cv::Point2d pp(P.at<double>(0,2),P.at<double>(1,2));
		float fx=P.at<double>(0,0);
		E = findEssentialMat(currentPts, previousPts, fx, pp, CV_RANSAC, 0.99, 1, motionInlierMask ); 
		recoverPose(E,currentPts,previousPts,outR,outT,fx,pp,motionInlierMask);
		//TODO calculate average scale for translation
		int totalAverageSamples=0;
		cv::Mat average=cv::Mat::zeros(3,1,CV_64F);//correctlyScaledTransform
		cv::Mat K=P(cv::Rect(0,0,3,3));

		for(int index=0;index<latestInter.currentInlierIndexes.size();index++)
		{

			if(motionInlierMask.at<bool>(0,index))
			{
				cv::Mat xnew,xold;
				//compute scale from projection 
				//projection pixel in previous frame
				xold=cv::Mat(3,1,CV_64F);
				xold.at<double>(0,0)=previous.inliers.at(latestInter.previousInlierIndexes.at(index)).distance.x;
				xold.at<double>(1,0)=previous.inliers.at(latestInter.previousInlierIndexes.at(index)).distance.y;
				xold.at<double>(2,0)=previous.inliers.at(latestInter.previousInlierIndexes.at(index)).distance.z;

				xnew=cv::Mat(3,1,CV_64F);
				xnew.at<double>(0,0)=current.inliers.at(latestInter.currentInlierIndexes.at(index)).distance.x;
				xnew.at<double>(1,0)=current.inliers.at(latestInter.currentInlierIndexes.at(index)).distance.y;
				xnew.at<double>(2,0)=current.inliers.at(latestInter.currentInlierIndexes.at(index)).distance.z;
				average+=((K.inv()*xold-outR*xnew)*outT.inv(cv::DECOMP_SVD))*outT;
				totalAverageSamples++;
				if(totalAverageSamples==15)
				{
					index=latestInter.currentInlierIndexes.size();
				}
			}
		}
		end=std::chrono::steady_clock::now();
		timeTaken=std::chrono::duration<double,std::milli>(end-start).count();
		latestInter.timings.push_back(timeTaken);
		latestInter.timingDescription.push_back("MotionExtraction");

	if(totalAverageSamples>0)
	{
		average=average/double(totalAverageSamples); 
		//store and send back output in a ros message
		for(int row=0;row<3;row++)
		{
			for(int column=0;column<3;column++)
			{
				latestInter.R[3*row +column]=outR.at<double>(row,column);
			}
		}
		latestInter.T[0]=average.at<double>(0,0);
		latestInter.T[1]=average.at<double>(1,0);
		latestInter.T[2]=average.at<double>(2,0);
	}
	else
	{

		std::cout<<"no Inliers detected"<<std::endl;
	}
		//populate messages 
		for(int index=0;index<latestInter.currentInlierIndexes.size();index++)
		{
			if(motionInlierMask.at<bool>(0,index))
			{
				latestInter.motionInlierMask.push_back(true);
			}
			else
			{
				latestInter.motionInlierMask.push_back(false);
			}
		}





		interFrame.push_back(latestInter);
		if(interFrame.size()>=nWindow-1)
		{
			interFrame.erase(interFrame.begin());
		}
	}

	debug++;
	publishCurrentState();

}

void WindowMatcher::publishCurrentState()
{
		front_end::Window message;
		for(int index=0;index<window.size();index++)
		{
			message.frames.push_back(window.at(index));
		}
		for(int index=0;index<interFrame.size();index++)
		{
			message.tracks.push_back(interFrame.at(index));
		}
		statePub.publish(message);
}









