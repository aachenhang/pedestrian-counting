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
#include "merge_location.h"
#include "constants_list.h"
#include "tiny-dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// learning convolutional neural networks (LeNet-5 like architecture)
void sample1_convnet(int minibatch_size = 10) {
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
		<< fully_connected_layer<tan_h>(120, 3);

	std::cout << "load models..." << std::endl;

	// load MNIST dataset
	std::vector<label_t> train_labels;
	std::vector<vec_t> train_images;

	load_mydataset(train_images, train_labels);
	/*
	cout << "watch train_images size" << endl;
	cout << train_images.size() << endl;

	cout << "watch train_labels size" << endl;
	cout << train_labels.size() << endl;

	cout << "watch train_images[0]" << endl;
	for (auto t : train_images[0]) {
	cout << t << endl;
	}
	cout << "watch train_labels[0]" << endl;
	cout << train_labels[0] << endl;*/

	std::cout << "start learning" << std::endl;

	progress_display disp(train_images.size());
	timer t;
	//int minibatch_size = 5000;

	optimizer.alpha *= std::sqrt(minibatch_size);

	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << t.elapsed() << "s elapsed." << std::endl;
		tiny_dnn::result res = nn.test(train_images, train_labels);
		std::cout << res.num_success << "/" << res.num_total << std::endl;

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() { disp += minibatch_size; };

	// training
	nn.train<mse>(optimizer, train_images, train_labels, minibatch_size, 20,
		on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;


	// save networks
	nn.save("LeNet-weights");
}


// convert image to vec_t
void convert_image(const cv::Mat& img,
	double scale,
	int w,
	int h,
	label_t label,
	std::vector<vec_t>& data,
	std::vector<label_t>& labels,
	double minv = -1.0,
	double maxv = 1.0)
{
	cv::Mat_<uint8_t> resized;
	cv::resize(img, resized, cv::Size(w, h));
	vec_t d;

	std::transform(resized.begin(), resized.end(), std::back_inserter(d),
		[=](uint8_t c) { return c * (maxv - minv) / 255.0 + minv; });
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

			/* Compute the mirror image *//*
										  Mat imgmirror = imMirror(img);
										  convert_image(imgmirror, 1.0, 32, 32, label, train_images, train_labels);*/
		}
		else {
			cout << "miss: " << stream.str() << endl;
		}
	}
}


void convnet_test(String imgFileName, double maxv, double minv) {
	/* Load the convolution network */
	network<sequential> nn;
	nn.load("LeNet-weights");

	/* convert imagefile to vec_t */
	vector<Rect> foundLocations;
	vector<Rect> res;

	Mat img = imread(imgFileName, IMREAD_GRAYSCALE);

	for (int i = 0; i < img.cols - 64; i += 8) {
		for (int j = 0; j < img.rows - 64; j += 8) {
			cv::Mat_<uint8_t> resized;
			vec_t d;
			Mat tmp = img(Rect(i, j, 64, 64));
			cv::resize(tmp, resized, cv::Size(32, 32));
			std::transform(resized.begin(), resized.end(), std::back_inserter(d),
				[=](uint8_t c) { return c * (maxv - minv) / 255.0 + minv; });
			auto prediction = nn.predict(d);

			cout << "cout prediction : ";
			for (double d : prediction)
				cout << d << "<<";
			cout << endl;

			label_t label = (prediction[0] > prediction[1] ? 0 : 1);

			if (label == 1) {
				foundLocations.push_back(Rect(i, j, 64, 64));
			}

		}
	}

	/* Merge the location */
	//res = mergeLocation(foundLocations);

	/* Display the rectangle */
	img = imread(imgFileName);
	Scalar GREEN = Scalar(0, 255, 0);
	for (int i = 0; i < foundLocations.size(); i++) {
		rectangle(img, foundLocations[i], GREEN, 2);
	}
	imshow("Result", img);
	waitKey(0);
	destroyAllWindows();
}