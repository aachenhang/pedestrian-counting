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



Mat imMirror(Mat img);
void computeDescriptor(vector<vector<float>> &alldescriptors,
					   String filepath,
					   int label,
	                   int numofsample,
					   HOGDescriptor &hog,
					   int fillflag = 0);
void hog_svm_save();
void hog_svm_load(HOGDescriptor &hog);
void hog_svm_detect();

