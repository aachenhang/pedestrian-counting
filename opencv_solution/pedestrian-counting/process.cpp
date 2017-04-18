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

#include "merge_location.h"
#include "constants_list.h"
#include "process.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;



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
						int fillflag) 
{
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
			mirrordescriptors.push_back(label);
			alldescriptors.push_back(mirrordescriptors);
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
	computeDescriptor(alldescriptors, CelebA_dataset_file, 1, CelebA_num, hog, FILL_ZERO);
	cout << "descriptors number: " << alldescriptors.size() << endl;*/

	/* Compute the negative samples */
	computeDescriptor(alldescriptors, negative_samples_file, -1, negative_num, hog);
	cout << "descriptors number: " << alldescriptors.size() << endl;

	/* Compute the positive hard samples */
	computeDescriptor(alldescriptors, positive_hard_samples_file, 1, positive_hard_num, hog);
	cout << "descriptors number: " << alldescriptors.size() << endl;

	/* Compute the negative hard samples */
	computeDescriptor(alldescriptors, negative_hard_samples_file, -1, negative_hard_num, hog);

	/* Set the featureMat and labelMat */
	int rows = alldescriptors.size();
	int cols = alldescriptors.begin()->size() - 1;
	Mat featureMat(rows, cols, CV_32FC1);
	Mat labelMat(rows, 1, CV_32SC1);
	for (int i = 0; i < rows; i++) {
		vector<float> descriptor = alldescriptors[i];
		for (int j = 0; j < descriptor.size() - 1; j++) {
			featureMat.at<float>(i, j) = descriptor[j];
		}
		labelMat.at<int>(i, 0) = descriptor[descriptor.size() - 1];	//Notice: Here the float is converted to int
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
		imwrite("Result.jpg", img);
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
	Mat mid = img(Rect(0, (rows - cols) / 2, cols, cols));
	imshow("mid", mid);
	resize(mid, img, Size(64, 64));
	imshow("dst", img);
	waitKey(0);
	destroyAllWindows();
}






void hog_svm_cnn_detect() {
	/* Load the hog detector */
	Size winsize(64, 64), blocksize(16, 16), blockstep(8, 8), cellsize(8, 8);
	int nbins = 9;
	HOGDescriptor hog(winsize, blocksize, blockstep, cellsize, nbins);
	hog_svm_load(hog);
	while (1) {
		/* Detect the image */
		Mat img = imread(image_test_file);
		Mat imgGrey = imread(image_test_file, IMREAD_GRAYSCALE);
		vector<Rect> foundLocations;
		double hitThreshold = 10.0;
		cout << "cin hitThreshold & finalThreshold" << endl;
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

		/* Let CNN verify the result */
		/* Load the convolution network */
		network<sequential> nn;
		nn.load("LeNet-weights");
		vector<vec_t> predictions;
		vector<Rect> candidates;
		for(Rect rect : res){
			cv::Mat_<uint8_t> resized;
			vec_t d;
			Mat tmp = imgGrey(rect);
			cv::resize(tmp, resized, cv::Size(64, 64));
			std::transform(resized.begin(), resized.end(), std::back_inserter(d),
				[=](uint8_t c) { return c * (1.0f - (-1.0f)) / 255.0 + (-1.0f); });
			predictions.push_back(nn.predict(d));
			cout << "cout prediction : ";
			for (double d : nn.predict(d))
				cout << d << "<<";
			cout << endl;
			candidates.push_back(rect);

		}
		while (1) {
			double diff = 0, pos = -1, neg = 1;
			int order;
			cout << "cin order" << endl;
			cin >> order;
			if (order == 0) {
				break;
			}
			if (order & 4) {
				cout << "cin diff" << endl;
				cin >> diff;
			}
			if (order & 2) {
				cout << "cin pos predict should greater than : " << endl;
				cin >> pos;
			}
			if (order & 1) {
				cout << "cin neg predict should less than :" << endl;
				cin >> neg;
			}

			/* convert imagefile to vec_t */
			vector<Rect> foundLocations;
			vector<Rect> notFoundLocations;

			for (int i = 0; i < predictions.size(); i++) {
				vec_t pre = predictions[i];
				if (pre[1] - pre[0] > diff && pre[1] > pos && pre[0] < neg) {
					foundLocations.push_back(candidates[i]);
				}
				else {
					notFoundLocations.push_back(candidates[i]);
				}
			}
			/* Display the rectangle */
			Scalar GREEN = Scalar(0, 255, 0);
			Scalar RED = Scalar(32, 85, 234);
			Mat imgSVM = img;
			for (int i = 0; i < res.size(); i++) {
				rectangle(imgSVM, res[i], GREEN, 2);
			}
			for (int i = 0; i < foundLocations.size(); i++) {
				rectangle(img, foundLocations[i], GREEN, 2);
			}
			for (int i = 0; i < notFoundLocations.size(); i++) {
				rectangle(img, notFoundLocations[i], RED, 2);
			}
			imshow("SVM", imgSVM);
			imshow("Result", img);
			cout << "SVM found:" << res.size() << endl;
			cout << "CNN found:" << foundLocations.size() << endl;
			waitKey(0);
			imwrite("Result.jpg", img);
			destroyAllWindows();
		}
	}
}


void computeDescriptor(vector<vector<float>> &alldescriptors,
					String filepath,
					int label,
					int numofsample,
					network<sequential> &nn,
					int fillflag) 
{
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
			Mat imgGrey = imread(stream.str(), IMREAD_GRAYSCALE);
			cv::Mat_<uint8_t> resized;
			if (fillflag == 1) {
				int cols = imgGrey.cols;
				int rows = imgGrey.rows;
				Mat mid = imgGrey(Rect(0, (rows - cols) / 2, cols, cols));
				resize(mid, imgGrey, Size(64, 64));
			}
			vec_t d;
			cv::resize(imgGrey, resized, cv::Size(64, 64));
			std::transform(resized.begin(), resized.end(), std::back_inserter(d),
				[=](uint8_t c) { return c * (1.0f - (-1.0f)) / 255.0 + (-1.0f); });
			nn.predict(d);
			for (float f : nn[nn.depth() - 2]->output().front().front()) {
				descriptors.push_back(f);
			}
			descriptors.push_back(label);
			alldescriptors.push_back(descriptors);

			/* Compute the mirror image */
			Mat imgmirror = imMirror(imgGrey);
			for (float f : nn[nn.depth() - 2]->output().front().front()) {
				mirrordescriptors.push_back(f);
			}
			mirrordescriptors.push_back(label);
			alldescriptors.push_back(mirrordescriptors);
		}
		else {
			cout << "miss: " << stream.str() << endl;
		}
	}
	
}


void svm_cnn_save() {
	/* Initial the time counter */
	time_t start = time(NULL);


	/* Initial the SVM */
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setC(0.01);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));

	/* Initial the CNN */
	network<sequential> nn;
	nn.load("LeNet-weights");

	/* Initial the descriptors */
	vector<vector<float>> alldescriptors;

	/* Compute the positive samples from my dataset */
	computeDescriptor(alldescriptors, positive_samples_file, 1, positive_num, nn);
	cout << "descriptors number: " << alldescriptors.size() << endl;


	/* Compute the positive samples from CelebA dataset *//*
	computeDescriptor(alldescriptors, CelebA_dataset_file, 1, CelebA_num, hog, FILL_ZERO);
	cout << "descriptors number: " << alldescriptors.size() << endl;*/

	/* Compute the negative samples */
	computeDescriptor(alldescriptors, negative_samples_file, -1, negative_num, nn);
	cout << "descriptors number: " << alldescriptors.size() << endl;

	/* Compute the positive hard samples */
	computeDescriptor(alldescriptors, positive_hard_samples_file, 1, positive_hard_num, nn);
	cout << "descriptors number: " << alldescriptors.size() << endl;

	/* Compute the negative hard samples */
	computeDescriptor(alldescriptors, negative_hard_samples_file, -1, negative_hard_num, nn);

	/* Set the featureMat and labelMat */
	int rows = alldescriptors.size();
	int cols = alldescriptors.begin()->size() - 1;
	Mat featureMat(rows, cols, CV_32FC1);
	Mat labelMat(rows, 1, CV_32SC1);
	for (int i = 0; i < rows; i++) {
		vector<float> descriptor = alldescriptors[i];
		for (int j = 0; j < descriptor.size() - 1; j++) {
			featureMat.at<float>(i, j) = descriptor[j];
		}
		labelMat.at<int>(i, 0) = descriptor[descriptor.size() - 1];	//Notice: Here the float is converted to int
	}

	/* Print the picture reading time */
	time_t end = time(NULL);
	cout << "cost: " << end - start << "s" << endl;

	/* Train the svm */
	cout << "Trainning starting..." << endl;
	start = time(NULL);
	svm->train(featureMat, ROW_SAMPLE, labelMat);
	svm->save(svm_cnn_file);
	end = time(NULL);
	cout << "cost: " << end - start << "s" << endl;
	cout << "save: " << svm_cnn_file << endl;
}


