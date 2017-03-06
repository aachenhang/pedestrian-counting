#ifndef TEST_H
#define TEST_H


#include <cstddef>
#include <ctime>
#include <iostream>


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
//
//#include <libvibe++/ViBe.h>
//#include <libvibe++/distances/Manhattan.h>
#include <stdint.h>
#include <mat.h>
#include <string>

const int maxofpoint = 10000;

using namespace std;
using namespace cv;

class myclass {
public:
	int add();
};

static __inline void mytest() {
	myclass m;
	printf("%d\n", m.add());
	return;
}

void inline plotdemo() {
	const char *matfilename = "F:/Downloads/Compressed/UCFCrowdCountingDataset_CVPR13/UCF_CC_50/2_ann.mat";
	const char *imgfilename = "F:/Downloads/Compressed/UCFCrowdCountingDataset_CVPR13/UCF_CC_50/2.jpg";
	MATFile* matfile = matOpen(matfilename, "r");
	mxArray *mxarr = matGetVariable(matfile, "annPoints");

	const size_t numofpoint = mxGetNumberOfElements(mxarr) / 2;


	double *data = (double *)mxGetData(mxarr);
	cout << data[0] << endl;
	cout << data[0] << endl;
	cout << data[99] << endl;
	cout << data[100] << endl;


	Mat img = imread(imgfilename);

	for (int i = 0; i < numofpoint; i++) {
		Scalar color = Scalar(0, 255, 0);
		rectangle(InputOutputArray(img), Point(data[i]-5, data[numofpoint+i] - 5), Point(data[i] + 5, data[numofpoint+i] + 5), color);
	}

	imshow("demo", img);
	cvWaitKey(0);

	Mat croppedimg = img(Rect(0, 0, 200, 200));
	imshow("croppedimg", croppedimg);
	cvWaitKey(0);

	matClose(matfile);
}


void inline getsizeofimg() {

	const char* imgfilename = "F:/Downloads/mydataset/0.jpg";
	Mat img = imread(imgfilename);
	cout << img.cols << endl;
}
#endif // !TEST_H