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


#include <opencv2\core\core.hpp>
#include <opencv2\objdetect\objdetect.hpp>
using namespace cv;

#include "merge_location.h"
#include "constants_list.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;



Mat imMirror(Mat img);
void computeDescriptor(vector<vector<float>> &alldescriptors,
					   String filepath,
					   int label,
	                   int numofsample,
					   HOGDescriptor &hog,
					   int fillflag = 0);
void hog_svm_save();
void hog_svm_load(HOGDescriptor &hog, String file = svm_file);
void hog_svm_detect();


void hog_svm_cnn_detect();

/* Pust the output weights of the last convolution layer of cnn to svm */
void computeDescriptor(vector<vector<float>> &alldescriptors,
	String filepath,
	int label,
	int numofsample,
	network<sequential> &nn,
	int fillflag = 0);
void svm_cnn_save();
void svm_cnn_detect();

