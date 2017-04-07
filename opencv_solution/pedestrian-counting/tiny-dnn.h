/*
Copyright (c) 2013, Taiga Nomi
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

using namespace std;

void sample1_convnet(int minibatch_size);
void load_mydataset(vector<vec_t>& train_images, vector<label_t>& train_labels);
void convert_image(const cv::Mat& img,
	double scale,
	int w,
	int h,
	label_t label,
	std::vector<vec_t>& data,
	std::vector<label_t>& labels,
	double minv,
	double maxv);
void load_sample(vector<vec_t>& train_images,
	vector<label_t>& train_labels,
	string filepath,
	int label,
	int numofsample,
	int fillflag = 0);
void convnet_test(String imgFileName = image_test_file, double maxv = 1.0, double minv = -1.0);
