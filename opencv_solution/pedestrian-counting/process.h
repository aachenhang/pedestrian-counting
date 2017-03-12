#pragma once

/**********************************************************
	This file contains the cascade about the all process.
/**********************************************************/


#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
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
String hard_samples_file = "F:/Downloads/mydataset/hard_sample/";
String CelebA_dataset_file = "E:/BaiduNetdiskDownload/CelebA/Img/img_align_celeba/";
//String svm_file = "F:/Downloads/mydataset/svm.xml";
String svm_file = "F:/Downloads/mydataset/svm_3000_25000.xml";
String image_test_file = "F:/Downloads/mydataset/35.jpg";
const int positive_num = 3000;
const int negative_num = 25000;
const int CelebA_num = 202599;
int mergeset[10005];


Mat imMirror(Mat img);
void computeDescriptor(vector<vector<float>> &alldescriptors,
					   String filepath,
					   int label,
	                   int numofsample,
					   HOGDescriptor &hog,
					   int fillflag);
void hog_svm_save();
void hog_svm_load(HOGDescriptor &hog);
void hog_svm_detect();
vector<Rect> mergeLocation(const vector<Rect> foundLocations);
int findRoot(int pos);
void merge(int a, int b);
bool canMerge(Rect a, Rect b);
bool cmp(Rect a, Rect b);


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


void computeDescriptor(vector<vector<float>> &alldescriptors,
					   String filepath,
					   int label,
					   int numofsample,
					   HOGDescriptor &hog,
					   int fillflag = 0) {
	
	for (int i = 0; i <= numofsample; i++) {
		vector<float> descriptors, mirrordescriptors;
		stringstream stream;
		if (fillflag == 1) {
			stream << filepath << setfill('0') << setw(6) << i << ".jpg";
		}
		else {
			stream << filepath << i << ".jpg";
		}
		ifstream f(stream.str());
		if (f.good()) {
			cout << "compute " << stream.str() << endl;
			Mat img = imread(stream.str());
			if (fillflag == 1) {
				int cols = img.cols;
				int rows = img.rows;
				Mat mid = img(Rect(0, (rows - cols) / 2, cols, cols));
				resize(mid, img, Size(64, 64));
			}
			hog.compute(img, descriptors);
			descriptors.push_back(label);
			alldescriptors.push_back(descriptors);
			
			/* Compute the mirror image */
			Mat imgmirror = imMirror(img);
			hog.compute(imgmirror, mirrordescriptors);
			descriptors.push_back(label);
			alldescriptors.push_back(descriptors);
		}
		else {
			cout << "miss: " << stream.str() << endl;
		}
	}
}



void hog_svm_save() {

	/* Initial the time counter */
	time_t start = time(NULL);


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
	
	/* Compute the positive samples from my dataset */
	computeDescriptor(alldescriptors, positive_samples_file, 1, positive_num, hog);
	cout << "descriptors number: " << alldescriptors.size() << endl;


	/* Compute the positive samples from CelebA dataset *//*
	computeDescriptor(alldescriptors, CelebA_dataset_file, 1, CelebA_num, hog, 1);
	cout << "descriptors number: " << alldescriptors.size() << endl;*/

	/* Compute the negative samples */
	computeDescriptor(alldescriptors, negative_samples_file, -1, negative_num, hog);
	cout << "descriptors number: " << alldescriptors.size() << endl;

	/* Set the featureMat and labelMat */
	int rows = alldescriptors.size();
	int cols = alldescriptors.begin()->size() - 1;
	Mat featureMat(rows, cols, CV_32FC1);
	Mat labelMat(rows, 1, CV_32SC1);
	for (int i = 0; i < rows; i++) {
		vector<float> descriptor = alldescriptors[i];
		for (int j = 0; j < descriptor.size()-1; j++) {
			featureMat.at<float>(i, j) = descriptor[j];
		}
		labelMat.at<int>(i, 0) = descriptor[descriptor.size()-1];	//Notice: Here the float is converted to int
	}

	/* Print the picture reading time */
	time_t end = time(NULL);
	cout << "cost: " << end - start << "s" << endl;

	/* Train the svm */
	cout << "Trainning starting..." << endl;
	start = time(NULL);
	svm->train(featureMat, ROW_SAMPLE, labelMat);
	svm->save(svm_file);
	end = time(NULL);
	cout << "cost: " << end - start << "s" << endl;
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


void hog_svm_detect() {
	
	/* Load the hog detector */
	Size winsize(64, 64), blocksize(16, 16), blockstep(8, 8), cellsize(8, 8);
	int nbins = 9;
	HOGDescriptor hog(winsize, blocksize, blockstep, cellsize, nbins);
	hog_svm_load(hog);
	while (1) {
		/* Detect the image */
		Mat img = imread(image_test_file);
		vector<Rect> foundLocations;
		double hitThreshold = 10.0;
		cin >> hitThreshold;
		if (hitThreshold == 0)	break;
		Size winStride(8, 8);
		Size padding(0, 0);
		double scale = 1.05;
		double finalThreshold = 100.0;
		cin >> finalThreshold;
		bool useMeanshiftGrouping = false;
		hog.detectMultiScale(img, foundLocations, hitThreshold, winStride, padding, finalThreshold, useMeanshiftGrouping);

		/* Merge the locations */
		vector<Rect> res = mergeLocation(foundLocations);

		/* Print the locations */
		sort(res.begin(), res.end(), cmp);
		for (Rect rect : res) {
			cout << rect.x << " " << rect.y << endl;
		}

		/* Display the rectangle */
		Scalar GREEN = Scalar(0, 255, 0);
		for (int i = 0; i < res.size(); i++) {
			rectangle(img, res[i], GREEN, 2);
		}
		imshow("Result", img);
		waitKey(0);
		destroyAllWindows();
	}
}


void testResize() {
	Mat img = imread(CelebA_dataset_file + "/000016.jpg");
	imshow("origin", img);
	int cols = img.cols;
	int rows = img.rows;
	cout << cols << endl;
	cout << rows << endl;
	Mat mid = img(Rect(0, (rows-cols)/2, cols, cols));
	imshow("mid", mid);
	resize(mid, img, Size(64, 64));
	imshow("dst", img);
	waitKey(0);
	destroyAllWindows();
}



vector<Rect> mergeLocation(const vector<Rect> foundLocations) {
	for (int i = 0; i < foundLocations.size(); i++) {
		mergeset[i] = -1;
		for (int j = 0; j < i; j++) {
			if (canMerge(foundLocations[i], foundLocations[j])) {
				merge(i, j);
			}
		}
	}
	vector<Rect> res;
	for (int i = 0; i < foundLocations.size(); i++) {
		int cnt = 0;
		Rect cur(0, 0, 64, 64);
		for (int j = 0; j < foundLocations.size(); j++) {
			if (findRoot(j) == i) {
				cnt++;
				cur.x += foundLocations[j].x;
				cur.y += foundLocations[j].y;
			}
		}
		if (cnt != 0) {
			cur.x /= cnt;
			cur.y /= cnt;
			res.push_back(cur);
		}
	}
	return res;
}


int findRoot(int pos) {
	return mergeset[pos] == -1 ? pos : mergeset[pos] = findRoot(mergeset[pos]);
}


void merge(int a, int b) {
	if (a > b)	swap(a, b);
	mergeset[b] = findRoot(a);
}


bool canMerge(Rect a, Rect b) {
	/* step size = 8, can be merge if not more than 2 steps */
	return abs(a.x - b.x) + abs(a.y - b.y) <= 8 * 2;
}


bool cmp(Rect a, Rect b) {
	return a.x < b.x || a.x == b.x && a.y < b.y;
}