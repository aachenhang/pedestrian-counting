#pragma once

/**********************************************************
	This file contains the cascade about the all process.
/**********************************************************/


#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
using namespace std;


#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;
using namespace ml;


String positive_samples_file = "F:/Downloads/mydataset/positive_sample/";
String negative_samples_file = "F:/Downloads/mydataset/negative_sample/";
String svm_file = "F:/Downloads/mydataset/svm.xml";
String image_test_file = "F:/Downloads/mydataset/34.jpg";
const int positive_num = 2018;
const int negative_num = 4050;


Mat imMirror(Mat img) {
	Mat dst(img.size(), img.type());
	Mat map_x(img.size(), CV_32FC1);
	Mat map_y(img.size(), CV_32FC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			map_x.at<float>(i, j) = img.cols - j;
			map_y.at<float>(i, j) = i;
		}
	}
	remap(img, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	return dst;
}


void computeDescriptor(vector<vector<float>> &alldescriptors, String filepath, int label, int numofsample, HOGDescriptor &hog) {
	
	for (int i = 0; i <= numofsample; i++) {
		vector<float> descriptors, mirrordescriptors;
		stringstream stream;
		stream << filepath << i << ".jpg";
		ifstream f(stream.str());
		if (f.good()) {
			cout << "compute " << stream.str() << endl;
			Mat img = imread(stream.str());
			hog.compute(img, descriptors);
			descriptors.push_back(label);
			alldescriptors.push_back(descriptors);

			/* Compute the mirror image */
			Mat imgmirror = imMirror(img);
			hog.compute(imgmirror, mirrordescriptors);
			descriptors.push_back(label);
			alldescriptors.push_back(descriptors);
		}
	}
}



void hog_svm_save() {

	/* Initial the SVM */
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setC(0.01);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));

	/* Initial the HOGDescriptor */
	Size winsize(64, 64), blocksize(16, 16), blockstep(8, 8), cellsize(8, 8);
	int nbins = 9;
	HOGDescriptor hog(winsize, blocksize, blockstep, cellsize, nbins);

	/* Initial the descriptors */
	vector<vector<float>> alldescriptors;
	
	/* Compute the positive samples */
	computeDescriptor(alldescriptors, positive_samples_file, 1, positive_num, hog);
	cout << "descriptors number: " << alldescriptors.size() << endl;

	/* Compute the negative samples */
	computeDescriptor(alldescriptors, negative_samples_file, -1, negative_num, hog);
	cout << "descriptors number: " << alldescriptors.size() << endl;

	/* Set the featureMat and labelMat */
	int rows = alldescriptors.size();
	int cols = alldescriptors.begin()->size();
	Mat featureMat(rows, cols, CV_32FC1);
	Mat labelMat(rows, 1, CV_32SC1);
	for (int i = 0; i < rows; i++) {
		vector<float> descriptor = alldescriptors[i];
		for (int j = 0; j < descriptor.size()-1; j++) {
			featureMat.at<float>(i, j) = descriptor[j];
		}
		labelMat.at<int>(i, 0) = descriptor[descriptor.size()-1];	//Notice: Here the float is converted to int
	}

	/* Train the svm */
	svm->train(featureMat, ROW_SAMPLE, labelMat);
	svm->save(svm_file);
	cout << "save: " << svm_file << endl;
}


void hog_svm_load(HOGDescriptor &hog) {

	/* Load the svm */
	Ptr<SVM> svm = SVM::load(svm_file);
	cout << "load: " << svm_file << endl;

	/* Get alpha and rho */
	Mat supportVector = svm->getSupportVectors();
	int alpharows = supportVector.rows;
	int alphacols = svm->getVarCount();
	Mat alpha(alpharows, alphacols, CV_32F);
	Mat svindex(1, alpharows, CV_64F);
	double rho = svm->getDecisionFunction(0, alpha, svindex);
	alpha.convertTo(alpha, CV_32F);

	/* Get the result Mat */
	Mat result = -1 * alpha * supportVector;

	/* Get the detector vector */
	vector<float> detector;
	CV_Assert(result.cols > 100);
	for (int i = 0; i < result.cols; i++) {
		detector.push_back(result.at<float>(0, i));
	}

	/* Load the hog detector */
	hog.setSVMDetector(detector);
	cout << "load: HOGDescriptor" << endl;
}


void hog_svm_detect(double hitThreshold, double finalThreshold) {
	
	/* Load the hog detector */
	Size winsize(64, 64), blocksize(16, 16), blockstep(8, 8), cellsize(8, 8);
	int nbins = 9;
	HOGDescriptor hog(winsize, blocksize, blockstep, cellsize, nbins);
	hog_svm_load(hog);

	/* Detect the image */
	Mat img = imread(image_test_file);
	vector<Rect> foundLoadcations;
	//double hitThreshold = 10.0;
	Size winStride(8, 8);
	Size padding(0, 0);
	double scale = 1.05;
	//double finalThreshold = 100.0;
	bool useMeanshiftGrouping = false;
	hog.detectMultiScale(img, foundLoadcations, hitThreshold, winStride, padding, finalThreshold, useMeanshiftGrouping);

	/* Display the rectangle */
	Scalar GREEN = Scalar(0, 255, 0);
	for (int i = 0; i < foundLoadcations.size(); i++) {
		rectangle(img, foundLoadcations[i], GREEN, 2);
	}
	imshow("Result", img);
	waitKey(0);
	destroyAllWindows();
}

