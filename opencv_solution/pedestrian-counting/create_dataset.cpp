#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
using namespace std;


#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>
using namespace cv;
using namespace ml;


#include "process.h"
#include "constants_list.h"

String imgfilename = "F:/Downloads/mydataset/1.jpg";
String videofilename = "F:/Downloads/8116_IP_segment_0.mp4";
String windowname = "makeAnnotation";
fstream f("F:/Downloads/mydataset/1.txt", ios::out | ios::in);
Scalar GREEN = Scalar(0, 255, 0);
const int rectanglesize = 64;

void mouseClick(int event, int x, int y, int flags, void* userdata) {
	Mat* imgptr = (Mat*)userdata;
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		if (rectanglesize / 2 <= x && x <= imgptr->cols - rectanglesize / 2 && rectanglesize / 2 <= y && y <= imgptr->rows - rectanglesize / 2) {
			rectangle(*imgptr, Rect(x - rectanglesize / 2, y - rectanglesize / 2, rectanglesize, rectanglesize), GREEN, 2);
			imshow(windowname, *imgptr);
			f << x << " " << y << endl;
		}
		break;
	default:
		break;
	}
}


int makeAnnotation(int inputnum) {

	/* initialize: imgfilename, f */
	stringstream tmpstream;
	tmpstream << "F:/Downloads/mydataset/" << inputnum << ".jpg";
	imgfilename = tmpstream.str();
	cout << imgfilename << endl;
	Mat img = imread(imgfilename);
	tmpstream.str("");
	tmpstream << "F:/Downloads/mydataset/" << inputnum << ".txt";
	f = fstream(tmpstream.str(), ios::out);

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
	destroyAllWindows();
	return 0;
}


int catchimage(int inputnum) {

	/* initialize: videofilename */
	stringstream tmpstream;
	tmpstream << "F:/Downloads/8116_IP_segment_" << inputnum << ".mp4";
	videofilename = tmpstream.str();
	VideoCapture cap(videofilename);

	Mat frame;
	stringstream stream;
	int num = 0, offset = 0;
	while (cap.read(frame)) {
		if (num % 264 == 0) {
			stream.str("");
			stream << "F:/Downloads/mydataset/" << offset << ".jpg";
			offset++;
			while (1) {
				ifstream infile(stream.str());
				if (infile.good()) {
					stream.str("");
					stream << "F:/Downloads/mydataset/" << offset << ".jpg";
					offset++;
					infile.close();
				}
				else {
					infile.close();
					break;
				}
			}
			imwrite(stream.str(), frame);
			cout << "write: " << stream.str() << endl;
		}
		num++;
	}

	return 0;
}


int cropimage(int inputnum) {

	/* initialize: imgfilename, f */
	stringstream tmpstream;
	tmpstream << "F:/Downloads/mydataset/" << inputnum << ".jpg";
	imgfilename = tmpstream.str();
	Mat img = imread(imgfilename);
	tmpstream.str("");
	tmpstream << "F:/Downloads/mydataset/" << inputnum << ".txt";
	f = fstream(tmpstream.str(), ios::out | ios::in);

	stringstream stream;
	int num = 0;
	int x, y;
	while (f >> x >> y) {
		Mat croppedimg = img(Rect(x - rectanglesize / 2, y - rectanglesize / 2, rectanglesize, rectanglesize));
		stream.str("");
		stream << "F:/Downloads/mydataset/positive_sample/" << num << ".jpg";
		num++;

		while (1) {
			ifstream infile(stream.str());
			if (infile.good()) {
				stream.str("");
				stream << "F:/Downloads/mydataset/positive_sample/" << num << ".jpg";
				num++;
				infile.close();
			}
			else {
				infile.close();
				break;
			}
		}
		imwrite(stream.str(), croppedimg);
		cout << "write: " << stream.str() << endl;
	}
	return 0;
}


int createNegativeSample(int inputnum, int sum) {

	/* initialize: imgfilename, f, srand */
	stringstream tmpstream;
	tmpstream << "F:/Downloads/mydataset/" << inputnum << ".jpg";
	imgfilename = tmpstream.str();
	Mat img = imread(imgfilename);
	tmpstream.str("");
	tmpstream << "F:/Downloads/mydataset/" << inputnum << ".txt";
	f = fstream(tmpstream.str(), ios::in);
	srand(time(NULL));

	stringstream stream;
	int x[505], y[505];
	int randx, randy;
	int numofpoint = 0;
	int num = 0;
	while (f >> x[numofpoint] >> y[numofpoint]) {
		numofpoint++;
	}
	cout << "readed " << numofpoint << " points" << endl;
	while (sum > 0) {

		/* generate x and y randomly */
		randx = rand() % (img.cols - rectanglesize + 1) + rectanglesize / 2;
		randy = rand() % (img.rows - rectanglesize + 1) + rectanglesize / 2;
		int i;
		for (i = 0; i < numofpoint; i++) {
			if ((randx - x[i])*(randx - x[i]) + (randy - y[i])*(randy - y[i]) < rectanglesize*rectanglesize) {
				break;
			}
		}
		if (i == numofpoint) {
			Mat croppedimg = img(Rect(randx - rectanglesize / 2, randy - rectanglesize / 2, rectanglesize, rectanglesize));
			stream.str("");
			stream << "F:/Downloads/mydataset/negative_sample/" << num << ".jpg";
			num++;

			while (1) {
				ifstream infile(stream.str());
				if (infile.good()) {
					stream.str("");
					stream << "F:/Downloads/mydataset/negative_sample/" << num << ".jpg";
					num++;
					infile.close();
				}
				else {
					infile.close();
					break;
				}
			}
			imwrite(stream.str(), croppedimg);
			cout << "write: " << stream.str() << endl;
			sum--;
		}

	}

	return 0;
}


void createHardSample() {

	/* Load the hog detector */
	Size winsize(64, 64), blocksize(16, 16), blockstep(8, 8), cellsize(8, 8);
	int nbins = 9;
	HOGDescriptor hog(winsize, blocksize, blockstep, cellsize, nbins);
	Ptr<SVM> svm = SVM::load(svm_file);
	cout << "load: " << svm_file << endl;

	/* Create the negative hard samples */
	int hardnum = 0;
	for (int i = 0; i < negative_num; i++) {
		stringstream stream;
		stream << negative_samples_file << i << ".jpg";
		if (ifstream(stream.str()).good()) {

			vector<float> descriptor;
			Mat mat = imread(stream.str());
			hog.compute(mat, descriptor);
			Mat featureMat(1, descriptor.size(), CV_32FC1);
			int cols = descriptor.size();
			for (int j = 0; j < cols; j++) {
				featureMat.at<float>(0, j) = descriptor[j];
			}
			if (svm->predict(featureMat) != -1) {
				cout << "Detected hard sample: " << stream.str() << endl;
				while (1) {
					stringstream hardstream;
					hardstream << negative_hard_samples_file << hardnum << ".jpg";
					if (ifstream(hardstream.str()).good()) {
						hardnum++;
					}
					else {
						imwrite(hardstream.str(), mat);
						cout << "create : " << hardstream.str() << endl;
						hardnum++;
						break;
					}
				}
			}
		}
	}

	/* Create the positive hard samples */
	hardnum = 0;
	for (int i = 0; i < positive_num; i++) {
		stringstream stream;
		stream << positive_samples_file << i << ".jpg";
		if (ifstream(stream.str()).good()) {

			vector<float> descriptor;
			Mat mat = imread(stream.str());
			hog.compute(mat, descriptor);
			Mat featureMat(1, descriptor.size(), CV_32FC1);
			int cols = descriptor.size();
			for (int j = 0; j < cols; j++) {
				featureMat.at<float>(0, j) = descriptor[j];
			}
			if (svm->predict(featureMat) != 1) {
				cout << "Detected hard sample: " << stream.str() << endl;
				while (1) {
					stringstream hardstream;
					hardstream << positive_hard_samples_file << hardnum << ".jpg";
					if (ifstream(hardstream.str()).good()) {
						hardnum++;
					}
					else {
						imwrite(hardstream.str(), mat);
						cout << "create : " << hardstream.str() << endl;
						hardnum++;
						break;
					}
				}
			}
		}
	}
}