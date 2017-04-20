#include "stdafx.h"

using namespace cv;
int mog2(char* video) {
	VideoCapture capvideo(video);
	Mat frame, mask, thresholdImage, output;

	int height = (int)capvideo.get(CV_CAP_PROP_FRAME_HEIGHT);
	int width = (int)capvideo.get(CV_CAP_PROP_FRAME_WIDTH);

	Mat bwframe(height, width, CV_8UC1);
	Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();

	while (capvideo.read(frame)) {
		cvtColor(frame, bwframe, CV_RGB2GRAY);
		mog2->apply(bwframe, mask, 0.01);
		//imshow("mask", mask);
		//imshow("Input video", bwframe);
		//waitKey(1);
	}

	capvideo.release();

	return 0;
}


int knnextract(char *video) {
	VideoCapture capvideo(video);
	Mat frame, mask, thresholdImage, output;

	int height = (int)capvideo.get(CV_CAP_PROP_FRAME_HEIGHT);
	int width = (int)capvideo.get(CV_CAP_PROP_FRAME_WIDTH);

	Mat bwframe(height, width, CV_8UC1);
	Ptr<BackgroundSubtractorKNN> knn = createBackgroundSubtractorKNN();

	while (capvideo.read(frame)) {
		cvtColor(frame, bwframe, CV_RGB2GRAY);
		knn->apply(bwframe, mask, 0.01);
		//imshow("mask", mask);
		//imshow("Input video", bwframe);
		//waitKey(1);
	}

	capvideo.release();

	return 0;
}

//#include <libvibe++/ViBe.h>
//#include <libvibe++/distances/Manhattan.h>
//#include <stdint.h>
////#include <libvibe++/system/types.h>
//
//using namespace std;
//using namespace ViBe;
//
//int Vibe(char* video) {
//
//	/* Parameterization of ViBe. */
//	typedef ViBeSequential<1, Manhattan<1> >	ViBe;
//
//	/* Random seed. */
//	srand(time(NULL));
//
//	cv::VideoCapture decoder(video);
//	cv::Mat frame;
//
//	int height = (int)decoder.get(CV_CAP_PROP_FRAME_HEIGHT);
//	int width = (int)decoder.get(CV_CAP_PROP_FRAME_WIDTH);
//
//	ViBe* vibe = NULL;
//	cv::Mat bwFrame(height, width, CV_8UC1);
//	cv::Mat segmentationMap(height, width, CV_8UC1);
//	bool firstFrame = true;
//
//	while (decoder.read(frame)) {
//		cv::cvtColor(frame, bwFrame, CV_RGB2GRAY);
//
//		if (firstFrame) {
//			/* Instantiation of ViBe. */
//			vibe = new ViBe(height, width, bwFrame.data);
//			firstFrame = false;
//		}
//
//		/* Segmentation and update. */
//		vibe->_CRTP_segmentation(bwFrame.data, segmentationMap.data);
//		vibe->_CRTP_update(bwFrame.data, segmentationMap.data);
//
//		/* Post-processing: 3x3 median filter. */
//		medianBlur(segmentationMap, segmentationMap, 3);
//
//		//imshow("Input video", bwFrame);
//		//imshow("Segmentation by ViBe", segmentationMap);
//
//		//cvWaitKey(1);
//	}
//
//	delete vibe;
//
//	cvDestroyAllWindows();
//	decoder.release();
//
//	return EXIT_SUCCESS;
//}
