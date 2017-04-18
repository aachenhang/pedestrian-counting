#pragma once
/*
Copyright (c) 2013, Taiga Nomi
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "tiny_dnn/tiny_dnn.h"
#include "process.h"
#include "merge_location.h"
#include "constants_list.h"
#include "tiny-dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////////////////////////
// learning convolutional neural networks (LeNet-5 like architecture)
void sample1_convnet(float alpha) {
	// construct LeNet-5 architecture
	network<sequential> nn;

	// load MNIST dataset
	std::vector<label_t> train_labels;
	std::vector<vec_t> train_images;

	load_mydataset(train_images, train_labels);
	adagrad optimizer;
	while (1) {
		int epoch;
		cout << "cin the alpha, original alpha is 0.01" << endl;
		cin >> alpha;
		cout << "cin the epoch, original epoch is 20" << endl;
		cin >> epoch;
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

		/*
			input : 1@64*64
			C1 :	3@60*60
			S2 :	3@30*30
			C3 :	9@26*26
			S4 :	9@13*13
			C5 :	27@9*9
			S6 :	27@3*3
			C7 :	91@1
			F8 :	2
		*/
		/*nn << convolutional_layer<tan_h>(64, 64, 5, 1, 3)
			<< average_pooling_layer<tan_h>(60, 60, 3, 2)
			<< convolutional_layer<tan_h>(30, 30, 5, 3, 9)
			<< average_pooling_layer<tan_h>(26, 26, 9, 2)
			<< convolutional_layer<tan_h>(13, 13, 5, 9, 27)
			<< average_pooling_layer<tan_h>(9, 9, 27, 3)
			<< convolutional_layer<tan_h>(3, 3, 3, 27, 91)
			<< fully_connected_layer<tan_h>(91, 2);*/


		/*
			epoch : 20
			alpha : 0.01
			input : 1@64*64
			C1 :	10@60*60
			S2 :	10@30*30
			C3 :	100@26*26
			S4 :	100@13*13
			C5 :	500@9*9
			S6 :	500@3*3
			C7 :	1000@1
			F8 :	2
		*//*
		nn << convolutional_layer<tan_h>(64, 64, 5, 1, 10)
			<< average_pooling_layer<tan_h>(60, 60, 10, 2)
			<< convolutional_layer<tan_h>(30, 30, 5, 10, 100)
			<< average_pooling_layer<tan_h>(26, 26, 100, 2)
			<< convolutional_layer<tan_h>(13, 13, 5, 100, 500)
			<< average_pooling_layer<tan_h>(9, 9, 500, 3)
			<< convolutional_layer<tan_h>(3, 3, 3, 500, 1000)
			<< fully_connected_layer<tan_h>(1000, 2);*/

		/*epoch: 20
		alpha : 0.01
		input : 1@64*64
		C1 :	5@60*60
		S2 :	5@30*30
		C3 :	25@26*26
		S4 :	25@13*13
		C5 :	125@9*9
		S6 :	125@3*3
		C7 :	125@1
		F8 : 2*/

		
		nn << convolutional_layer<tan_h>(64, 64, 5, 1, 3)
		<< average_pooling_layer<tan_h>(60, 60, 3, 2)
		<< convolutional_layer<tan_h>(30, 30, 5, 3, 9)
		<< average_pooling_layer<tan_h>(26, 26, 9, 2)
		<< convolutional_layer<tan_h>(13, 13, 5, 9, 27)
		<< average_pooling_layer<tan_h>(9, 9, 27, 3)
		<< convolutional_layer<tan_h>(3, 3, 3, 27, 54)
		<< fully_connected_layer<tan_h>(54, 2);

		std::cout << "load models..." << std::endl;

		std::cout << "start learning" << std::endl;

		progress_display disp(train_images.size());
		timer t;
		int minibatch_size = 10;

		optimizer.alpha = alpha;

		// create callback
		auto on_enumerate_epoch = [&]() {
			std::cout << t.elapsed() << "s elapsed." << std::endl;
			/*tiny_dnn::result res = nn.test(train_images, train_labels);
			std::cout << res.num_success << "/" << res.num_total << std::endl;*/

			nn.save("LeNet-weights");
			cout << "The network was saved." << endl;

			disp.restart(train_images.size());
			t.restart();
		};

		auto on_enumerate_minibatch = [&]() { disp += minibatch_size; };

		// training
		nn.train<mse>(optimizer, train_images, train_labels, minibatch_size, epoch,
			on_enumerate_minibatch, on_enumerate_epoch);

		std::cout << "end training." << std::endl;


		// save networks
		nn.save("LeNet-weights");
	}
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
			cout << "compute " << stream.str() << "\r";
			Mat img = imread(stream.str(), IMREAD_GRAYSCALE);

			/* CelebA's size is 178*218, different from mydataset's 64*64 */
			if (fillflag == 1) {
				int cols = img.cols;
				int rows = img.rows;
				Mat mid = img(Rect(0, (rows - cols) / 2, cols, cols));
				resize(mid, img, Size(64, 64));
			}

			convert_image(img, 1.0, 64, 64, label, train_images, train_labels);

			/* Compute the mirror image *//*
										  Mat imgmirror = imMirror(img);
										  convert_image(imgmirror, 1.0, 32, 32, label, train_images, train_labels);*/
		}
		else {
			cout << "miss: " << stream.str() << "\r";
		}
	}
	cout << endl;
}


void convnet_test(String imgFileName, double maxv, double minv) {
	/* Load the convolution network */
	network<sequential> nn;
	nn.load("LeNet-weights");
	Mat img = imread(imgFileName, IMREAD_GRAYSCALE);
	vector<vec_t> predictions;
	vector<Rect> candidates;
	for (int i = 0; i < img.cols - 64; i += 8) {
		for (int j = 0; j < img.rows - 64; j += 8) {
			cv::Mat_<uint8_t> resized;
			vec_t d;
			Mat tmp = img(Rect(i, j, 64, 64));
			cv::resize(tmp, resized, cv::Size(64, 64));
			std::transform(resized.begin(), resized.end(), std::back_inserter(d),
				[=](uint8_t c) { return c * (maxv - minv) / 255.0 + minv; });
			predictions.push_back(nn.predict(d));
			if (i == 0) {
				cout << "cout prediction : ";
				for (double d : nn.predict(d))
					cout << d << "<<";
				cout << endl;
			}
			candidates.push_back(Rect(i, j, 64, 64));

		}
	}
	while (1) {
		double diff = 0, pos = -1, neg = 1;
		int order;
		cout << "cin order" << endl;
		cin >> order;
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
		vector<Rect> res;

		for (int i = 0; i < predictions.size(); i++) {
			vec_t pre = predictions[i];
			if (pre[1] - pre[0] > diff && pre[1] > pos && pre[0] < neg) {
				foundLocations.push_back(candidates[i]);
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
}