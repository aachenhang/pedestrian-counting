// experiment.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "getthreshold.h"
using namespace std;
using namespace cv;

void mouseClick(int event, int x, int y, int flags, void* userdata) {
	Mat img = *(Mat*)userdata;
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		cout << "at " << x << ", " << y << endl;
		cout << (int)img.at<unsigned char>(y, x) << endl;
		break;
	default:
		break;
	}
}


int main()
{
	int idx = 0;
	while (1) {
		String testFile = positive_samples_file + to_string(idx) + ".jpg";
		idx++;
		Mat imgOrigin = imread(testFile);
		Mat imgGrey = imread(testFile, IMREAD_GRAYSCALE);
		Mat img = imread(testFile, IMREAD_GRAYSCALE);
	
		vector<int> HistGram(256);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				HistGram[img.at<unsigned char>(i, j)]++;
			}
		}

		int threshold = Thre::GetMeanThreshold(HistGram);
		cout << "threshold = " << threshold << endl;

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				img.at<unsigned char>(i, j) = (img.at<unsigned char>(i, j) > threshold ? 255 : 0);
			}
		}

		imshow("origin", imgOrigin);
		imshow("imgGrey", imgGrey);
		imshow("binaryzation", img);
		moveWindow("origin", 900, 500);
		moveWindow("imgGrey", 900, 700);
		moveWindow("binaryzation", 900, 900);
		setMouseCallback("imgGrey", mouseClick, &imgGrey);

		waitKey(0);
		destroyAllWindows();
	}
    return 0;
}

