#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <dc1394/dc1394.h>

#include <inttypes.h>

#include <vector>
#include <memory.h>
#include <queue>
#include <mutex>
#include <pthread.h>

#define _FILE_OFFSET_BITS 64
#include <fstream>

using namespace std;

mutex bufferMutex;
mutex finishedMutex;

bool Finished=true;

int imageTotal=120;

void *dcBuffer,*threadBuffer;


mutex writeMutex;


void *writePtr;
void *readPtr;

int nBuffers=15*10;
int bytePerImage=2*768*1024;


char currentDate[64];




void *copyThread(void *threadId)
{
		ofstream logFile;
    void *lastAddress=threadBuffer+nBuffers*bytePerImage;//the final byte in memory
		unsigned long totalCopied=0;
		char fileName[100];
		char logname[100];
		sprintf(fileName,"/home/ubuntu/FLASHDISK/%s.rawData",currentDate);
		sprintf(logname,"/home/ubuntu/FLASHDISK/%s.cpLog",currentDate);
		logFile.open(logname);


		logFile<<"totalImages,nBytes"<<endl;

    FILE *f;
		f=fopen(fileName,"w");
    printf("ThreadStarted\n");
    void *lastPtr;


    while(true)
    {
        unique_lock<mutex> mlock(writeMutex);
        lastPtr=writePtr;
        mlock.unlock();
        if(lastPtr!=readPtr)
        {
						int totalImageBytes=0;
            if(lastPtr>readPtr)
            {
                //normal Copy
                int  nBytes=((char *)lastPtr-(char *)readPtr)*sizeof(char);
                fwrite(readPtr,1,nBytes,f);
								totalImageBytes=nBytes;
            }
            else
            {
                int firstnBytes=((char *) lastAddress-(char *)readPtr)*sizeof(char);
                int remainingBytes=((char *)lastPtr - (char *)threadBuffer)*sizeof(char);
								fwrite(readPtr,1,firstnBytes,f);
								fwrite(threadBuffer,1,remainingBytes,f);
								totalCopied+=(unsigned long) (firstnBytes+remainingBytes)/bytePerImage;
								totalImageBytes=firstnBytes+remainingBytes;
            }
						logFile<<hex<<(uint64_t)readPtr<<dec<<","<<hex<<(uint64_t)lastPtr<<dec<<","<<totalImageBytes<<endl;

            readPtr=lastPtr;
        }
        else
        {
            usleep(200);
        }
    }
		fflush(f);
    fclose(f);
		logFile.flush();
		logFile.close();
		
    printf("ThreadReturning\n");
    pthread_exit(NULL);
}


void cleanup_and_exit(dc1394camera_t *camera)
{
    dc1394_video_set_transmission(camera, DC1394_OFF);
    dc1394_capture_stop(camera);
    dc1394_camera_free(camera);
}

int main()
{
	time_t t=time(NULL);
	struct tm *tm=localtime(&t);
	strftime(currentDate,sizeof(currentDate),"%d-%m-%Y--%H-%M-%S",tm);

	ofstream logFile;
	
	char fileName[100];
	sprintf(fileName,"/home/ubuntu/FLASHDISK/%s.LoggedData",currentDate);
	logFile.open(fileName);
	logFile<<"stamp,jitter,bufferIndex,total\n";

	unsigned long totalImages=0;


	dcBuffer=malloc(nBuffers*bytePerImage);
  threadBuffer=dcBuffer;
  writePtr=threadBuffer;
  readPtr=writePtr;

  dc1394camera_t *camera;
	dc1394video_frame_t *frame;
	dc1394camera_list_t * list;
   dc1394error_t err;

	dc1394_t *d;

  pthread_t testThread;
  pthread_create(&testThread,NULL,copyThread,NULL);
  usleep(1000*1000);
	d = dc1394_new ();
  if (!d)
	{
  	return 1;
	}
  err=dc1394_camera_enumerate (d, &list);
  DC1394_ERR_RTN(err,"Failed to enumerate cameras");

  if (list->num == 0) {
  	dc1394_log_error("No cameras found");
  	return 1;
  }

  camera = dc1394_camera_new (d, list->ids[0].guid);
  if (!camera)
	{
      return 1;
  }
  dc1394_camera_free_list (list);


	clock_t startClock=clock();
  

  err=dc1394_video_set_iso_speed(camera,DC1394_ISO_SPEED_400);
  dc1394_video_set_framerate(camera,DC1394_FRAMERATE_15);
  dc1394_capture_setup(camera,50,DC1394_CAPTURE_FLAGS_DEFAULT);

  dc1394video_mode_t video_mode;
  dc1394_video_get_mode(camera,&video_mode);
  uint32_t width,height;
  dc1394_get_image_size_from_video_mode(camera,video_mode,&width,&height);

	err=dc1394_video_set_transmission(camera, DC1394_ON);
  DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not start camera iso transmission");
  int prevStamp=0;
	void * prevWriteAddress=threadBuffer;
	dc1394video_frame_t stereo_frame;
	printf("Beginning Stream @ %s\n",currentDate);
	printf("Buffer Start and End Pointers[%p,%p]\n",dcBuffer,dcBuffer+nBuffers*bytePerImage);
		
	while(true)
	{
		for(int bufferIndex=0;bufferIndex<nBuffers;bufferIndex++)
		{
				err=dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT, &frame);
   	    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not capture a frame");

        memcpy(&stereo_frame,frame,sizeof(dc1394video_frame_t)); //copy MetaData
	      stereo_frame.allocated_image_bytes=0;
	      stereo_frame.image=NULL;
        dc1394_deinterlace_stereo_frames(frame,&stereo_frame,DC1394_STEREO_METHOD_INTERLACED);
				dc1394_capture_enqueue(camera,frame);
        int diff=(int)(stereo_frame.timestamp-prevStamp);
        prevStamp=(int)stereo_frame.timestamp;
        void *currentWriteAddress=dcBuffer+bufferIndex*bytePerImage;
        memcpy(currentWriteAddress,stereo_frame.image,bytePerImage);
        unique_lock<mutex> mlock(writeMutex);
        writePtr=currentWriteAddress;
        mlock.unlock();
				free(stereo_frame.image);
				
				totalImages++;
				logFile<<dec<<stereo_frame.timestamp<<","<<diff<<","<<stereo_frame.id<<","<<totalImages;
				logFile<<","<<hex<<(uint64_t)currentWriteAddress<<dec<<","<<hex<<((uint64_t)currentWriteAddress-(uint64_t)prevWriteAddress)<<endl;
				prevWriteAddress=currentWriteAddress;
		}
		
	}
	printf("Finished\n");

    //wait for the second thread to finish 
    if(pthread_join(testThread, NULL)) {

        printf("asgsagasgas\n");
        usleep(100000);
        return 2;

    }

    printf("TotalThread\n");



    dc1394_video_set_transmission(camera, DC1394_OFF);
    dc1394_capture_stop(camera);
    dc1394_camera_free(camera);
    dc1394_free (d);

    free(dcBuffer);
		logFile.close();
	return 0;
}

   // dc1394_capture_enqueue(camera,frame);

   // dc1394video_frame_t stereo_frame;
	 // memcpy(&stereo_frame,frame,sizeof(dc1394video_frame_t)); //copy MetaData
	 // stereo_frame.allocated_image_bytes=0;
	//  stereo_frame.image=NULL;
   // dc1394_deinterlace_stereo_frames(frame,&stereo_frame,DC1394_STEREO_METHOD_INTERLACED);
   // d_pt=(short int*)&bayerImage.data[0];
   //     memcpy(d_pt,stereo_frame.image,1536*1024);
    //cv::imshow("a",bayerImage);
    //cv::waitKey(1);



		//printf("TimeStamp,%f,frames_behind,%d,FrameIDBuffer,%d,DCStamp,%d\n",timestamp,(int)frame->frames_behind,(int)frame->id,((int)frame->timestamp)-prevStamp);
    /*char fileName[50];
    sprintf(fileName,"/home/ubuntu/SD/%d.pgm",(int)frame->timestamp);
    printf("%d,%d\n",(int)frame->frames_behind,((int)frame->timestamp)-prevStamp);
    prevStamp=(int)frame->timestamp;
    imagefile=fopen(fileName, "wb");
    fprintf(imagefile,"P5\n%u %u 255\n",width,2*height);
    fwrite(stereo_frame.image, 1, 2*height*width, imagefile);
    fclose(imagefile);*/


	    /*-----------------------------------------------------------------------
     *  save image as 'Image.pgm'
     *-----------------------------------------------------------------------*/
   /* imagefile=fopen(IMAGE_FILE_NAME, "wb");

    if( imagefile == NULL) {
        perror( "Can't create '" IMAGE_FILE_NAME "'");
        cleanup_and_exit(camera);
    }

    dc1394_get_image_size_from_video_mode(camera, video_mode, &width, &height);
    fprintf(imagefile,"P5\n%u %u 255\n", width, height);
    fwrite(frame->image, 1, height*width, imagefile);
    fclose(imagefile);
    printf("wrote: " IMAGE_FILE_NAME "\n");*/ 
