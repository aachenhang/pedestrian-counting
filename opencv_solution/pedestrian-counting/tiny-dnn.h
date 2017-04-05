#pragma once
/*
Copyright (c) 2013, Taiga Nomi
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include "tiny_dnn/tiny_dnn.h"
#include "process.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

using namespace std;

void sample1_convnet();
void load_mydataset(vector<vec_t>& train_images, vector<label_t>& train_labels);
void convert_image(const cv::Mat& img,
	double scale,
	int w,
	int h,
	label_t label,
	std::vector<vec_t>& data,
	std::vector<label_t>& labels);
void load_sample(vector<vec_t>& train_images,
	vector<label_t>& train_labels,
	string filepath,
	int label,
	int numofsample,
	int fillflag = 0);



///////////////////////////////////////////////////////////////////////////////
// learning convolutional neural networks (LeNet-5 like architecture)
void sample1_convnet() {
	// construct LeNet-5 architecture
	network<sequential> nn;
	adagrad optimizer;

	// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
	// clang-format off
	static const bool connection[] = {
		O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
		O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
		O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
		X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
		X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
		X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
	};
	// clang-format on
#undef O
#undef X

	nn << convolutional_layer<tan_h>(32, 32, 5, 1,
		6) /* 32x32 in, 5x5 kernel, 1-6 fmaps conv */
		<< average_pooling_layer<tan_h>(28, 28, 6,
			2) /* 28x28 in, 6 fmaps, 2x2 subsampling */
		<< convolutional_layer<tan_h>(14, 14, 5, 6, 16,
			connection_table(connection, 6, 16))
		<< average_pooling_layer<tan_h>(10, 10, 16, 2)
		<< convolutional_layer<tan_h>(5, 5, 5, 16, 120)
		<< fully_connected_layer<tan_h>(120, 2);

	std::cout << "load models..." << std::endl;

	// load MNIST dataset
	std::vector<label_t> train_labels;
	std::vector<vec_t> train_images;

	load_mydataset(train_images, train_labels);

	std::cout << "start learning" << std::endl;

	progress_display disp(train_images.size());
	timer t;
	int minibatch_size = 10;

	optimizer.alpha *= std::sqrt(minibatch_size);

	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << t.elapsed() << "s elapsed." << std::endl;

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() { disp += minibatch_size; };

	// training
	nn.train<mse>(optimizer, train_images, train_labels, minibatch_size, 20,
		on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;


	// save networks
	std::ofstream ofs("LeNet-weights");
	ofs << nn;
}


// convert image to vec_t
void convert_image(const cv::Mat& img,
	double scale,
	int w,
	int h,
	label_t label,
	std::vector<vec_t>& data,
	std::vector<label_t>& labels)
{
	cv::Mat_<uint8_t> resized;
	cv::resize(img, resized, cv::Size(w, h));
	vec_t d;

	std::transform(resized.begin(), resized.end(), std::back_inserter(d),
		[=](uint8_t c) { return c * scale; });
	data.push_back(d);
	labels.push_back(label);
}


void load_mydataset(vector<vec_t>& train_images, vector<label_t>& train_labels) {
	/* Compute the positive samples from my dataset */
	load_sample(train_images, train_labels, positive_samples_file, 1, positive_num);
	cout << "images number: " << train_images.size() << endl;


	/* Compute the positive samples from CelebA dataset *//*
	load_sample(train_images, train_labels, CelebA_dataset_file, 1, CelebA_num, FILL_ZERO);
	cout << "images number: " << train_images.size() << endl;*/

	/* Compute the negative samples */
	load_sample(train_images, train_labels, negative_samples_file, 0, negative_num);
	cout << "images number: " << train_images.size() << endl;

	/* Compute the positive hard samples */
	load_sample(train_images, train_labels, positive_hard_samples_file, 1, positive_hard_num);
	cout << "images number: " << train_images.size() << endl;

	/* Compute the negative hard samples */
	load_sample(train_images, train_labels, negative_hard_samples_file, 0, negative_hard_num);
	cout << "images number: " << train_images.size() << endl;
}


void load_sample(vector<vec_t>& train_images, 
	vector<label_t>& train_labels, 
	string filepath,
	int label,
	int numofsample,
	int fillflag) {

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
			Mat img = imread(stream.str(), IMREAD_GRAYSCALE);

			/* CelebA's size is 178*218, different from mydataset's 64*64 */
			if (fillflag == 1) {
				int cols = img.cols;
				int rows = img.rows;
				Mat mid = img(Rect(0, (rows - cols) / 2, cols, cols));
				resize(mid, img, Size(64, 64));
			}

			convert_image(img, 1.0, 32, 32, label, train_images, train_labels);

			/* Compute the mirror image */
			Mat imgmirror = imMirror(img);
			convert_image(imgmirror, 1.0, 32, 32, label, train_images, train_labels);
		}
		else {
			cout << "miss: " << stream.str() << endl;
		}
	}
}