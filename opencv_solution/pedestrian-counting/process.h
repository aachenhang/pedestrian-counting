#pragma once

/**********************************************************
	This file contains the cascade about the all process.
/**********************************************************/





#include "constants_list.h"




Mat imMirror(Mat img);

void hog_svm_save();
void hog_svm_detect();


void hog_svm_cnn_detect();

/* Pust the output weights of the last convolution layer of cnn to svm */

void svm_cnn_save();
void svm_cnn_detect();

void svm_lbp_detect();