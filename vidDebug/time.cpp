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

using namespace std;

queue<char> imageBuffer;
mutex bufferMutex;
mutex finishedMutex;

bool Finished=true;

int imageTotal=5000;



void *copyThread(void *threadId)
{
    FILE *f;
	f=fopen("/home/ubuntu/FLASHDISK/testOutput.info","w");
    int totalBytes=imageTotal*2*768*1024;
    printf("ThreadStarted\n");
    while((totalBytes>0))
    {

        unique_lock<mutex> mlock(bufferMutex);
        if(imageBuffer.size()>0)
        {

            fputc(imageBuffer.front(),f);
            imageBuffer.pop();
            totalBytes-=1;
        }
        else
        {
            usleep(1000);
        }
        mlock.unlock();
    }
    fclose(f);
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
	int index=0;

	FILE* imagefile;
    dc1394camera_t *camera;
	dc1394video_frame_t *frame;
	dc1394camera_list_t * list;

    dc1394error_t err;
    int i;
	dc1394_t *d;

    pthread_t testThread;
    pthread_create(&testThread,NULL,copyThread,NULL);
	d = dc1394_new ();
    if (!d)
        return 1;
    err=dc1394_camera_enumerate (d, &list);
    DC1394_ERR_RTN(err,"Failed to enumerate cameras");

    if (list->num == 0) {
        dc1394_log_error("No cameras found");
        return 1;
    }

    camera = dc1394_camera_new (d, list->ids[0].guid);
    if (!camera) {
        dc1394_log_error("Failed to initialize camera with guid %"PRIx64, list->ids[0].guid);
        return 1;
    }
    dc1394_camera_free_list (list);


	clock_t startClock=clock();
    printf("%"PRIx64"\n",camera->guid);
  

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

	for(index;index<imageTotal;index++)
	{

        dc1394video_frame_t stereo_frame;
		err=dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT, &frame);
   	    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not capture a frame");

        memcpy(&stereo_frame,frame,sizeof(dc1394video_frame_t)); //copy MetaData
        int diff=(int)(frame->timestamp-prevStamp);
        prevStamp=(int)frame->timestamp;
        int index=0;
        unique_lock<mutex> mlock(bufferMutex);

        for(index;index<768*2*1024;index++)
	    {
            imageBuffer.push((char)stereo_frame.image[index]);
	    }
        mlock.unlock();
	    dc1394_capture_enqueue(camera,frame);
        printf("%d-%d\n",index,diff);

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