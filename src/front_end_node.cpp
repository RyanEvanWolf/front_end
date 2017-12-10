#include <ros/ros.h>




#include <front_end/StereoCamera.hpp>


stereo::StereoRect bumbleCamera;
cv::Mat mainImage;
cv::Mat leftImage,rightImage;
cv::Mat leftROI,rightROI;

sensor_msgs::ImagePtr messageLroi,messageRroi;
sensor_msgs::ImagePtr messagecolour,messageL,messageR;


#define DEFAULT_RECTIFICATION_FILE "/home/ubuntu/ConfigurationFiles/stereo_ParameterFive.xml"
image_transport::Publisher *pubL,*pubR,*pubLroi,*pubRroi,*pubCol;


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
		cv::cvtColor(cv_bridge::toCvShare(msg, "8UC1")->image,mainImage,cv::COLOR_BayerBG2RGB);
		//cv::imshow("out",mainImage);
		//cv::waitKey(1);
		cv::Mat grey;
		cv::cvtColor(mainImage,grey,CV_RGB2GRAY);
		
		rightImage=grey(cv::Rect(0,0,1024,768));
		cv::remap(rightImage,rightImage,bumbleCamera.R_fMapx_,bumbleCamera.R_fMapy_,cv::INTER_LINEAR);
		rightROI=rightImage(bumbleCamera.r_ROI_);

		leftImage=grey(cv::Rect(0,768,1024,768));
		cv::remap(leftImage,leftImage,bumbleCamera.L_fMapx_,bumbleCamera.L_fMapy_,cv::INTER_LINEAR);
		leftROI=leftImage(bumbleCamera.l_ROI_);


		messageLroi=cv_bridge::CvImage(std_msgs::Header(),"8UC1",leftROI).toImageMsg();
		messageRroi=cv_bridge::CvImage(std_msgs::Header(),"8UC1",rightROI).toImageMsg();
		messageL=cv_bridge::CvImage(std_msgs::Header(),"8UC1",leftImage).toImageMsg();
		messageR=cv_bridge::CvImage(std_msgs::Header(),"8UC1",rightImage).toImageMsg();
		//messagecolour=cv_bridge::CvImage(std_msgs::Header(),"8UC3",mainImage).toImageMsg();		

		pubLroi->publish(messageLroi);
		pubRroi->publish(messageRroi);
		pubL->publish(messageL);
		pubR->publish(messageR);
		//pubCol->publish(messagecolour);

  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc,char *argv[])
{
	ros::init(argc,argv,"front_end");
	ros::NodeHandle n;

	stereo::StereoCamera bumble(DEFAULT_RECTIFICATION_FILE);
	std::cout<<bumble.cameraSettings_.R_<<std::endl;
	/*cv::FileStorage in(DEFAULT_RECTIFICATION_FILE,cv::FileStorage::READ);
	in["StereoRect"]>>bumbleCamera;
	in.release();


 	image_transport::ImageTransport it(n);
 	image_transport::Subscriber sub = it.subscribe("dataset/currentImage", 5, imageCallback);
	image_transport::Publisher  leftRectifiedPub=it.advertise("bumblebee/leftRectified",5);
	image_transport::Publisher 	rightRectifiedPub=it.advertise("bumblebee/rightRectified",5);
	image_transport::Publisher  leftROIPub=it.advertise("bumblebee/leftROI",5);
	image_transport::Publisher 	rightROIPub=it.advertise("bumblebee/rightROI",5);
	image_transport::Publisher 	colourpub=it.advertise("bumblebee/colour",5);

	pubL = &leftRectifiedPub;
	pubR = &rightRectifiedPub;
	pubLroi = &leftROIPub;
	pubRroi= &rightROIPub;
	pubCol = &colourpub;

	*/
	ros::spin();
	
	return 0;
}
