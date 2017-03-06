#pragma once

#include <iostream>
#include <fstream>
using namespace std;


#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;

const char* imgfilename = "F:/Downloads/mydataset/0.jpg";
const char* windowname = "makeAnnotation";
fstream f("F:/Downloads/mydataset/tmp.txt", ios::out);
Scalar GREEN = Scalar(0, 255, 0);
const int rectanglesize = 50;

void mouseClick(int event, int x, int y, int flags, void* userdata) {
	Mat* imgptr = (Mat*)userdata;
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		rectangle(*imgptr, Rect(x - rectanglesize/2, y - rectanglesize / 2, rectanglesize, rectanglesize), GREEN, 2);
		imshow(windowname, *imgptr);
		f << x << " " << y << endl;
		break;
	default:
		break;
	}
}


int makeAnnotation() {
	Mat img = imread(imgfilename);
	char c;
	namedWindow(windowname);
	setMouseCallback(windowname, mouseClick, &img);
	while (1) {
		imshow(windowname, img);
		c = waitKey(0);

		if (c == 'q') {
			break;
		}
	}
	f.close();

	return 0;
}


