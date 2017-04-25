// cnntrain.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;
using namespace std;


static void parse_mydataset_train_data_labels(vector<vec_t> &train_data,
									vector<label_t> &train_labels) {
	/* Load allData & allLabel from the txt */
	Mat allData, allLabel;
	FileStorage file(TINY_ALL_IMAGE_FILE, FileStorage::READ);
	file["allData"] >> allData;
	file["allLabel"] >> allLabel;
	file.release();
	cout << "allData.size() = " << allData.size() << endl;
	cout << "allLabel.size() = " << allLabel.size() << endl;

	for (int i = 0; i < allData.rows; i++) {
		vec_t vec(allData.cols);
		for (int j = 0; j < allData.cols; j++) {
			vec[j] = static_cast<float>(allData.at<uint8_t>(i, j));
		}
		train_data.push_back(vec);
		train_labels.push_back(label_t(allLabel.at<int>(i, 0) == 1 ? 1 : 0));

		/* Add mirror data */
		if (allLabel.at<int>(0,i) == 1) {
			vec_t vecMirror(allData.cols);
			for (int j = 0; j < allData.cols; j++) {
				vecMirror[j] = static_cast<float>(allData.at<uint8_t>(i, j-j%64*2+63));
			}
			train_data.push_back(vecMirror);
			train_labels.push_back(label_t(1));
		}
	}

}

static void generate_test_data(vector<vec_t> &train_data,
							vector<label_t> &train_labels,
							vector<vec_t> &test_data,
							vector<label_t> &test_labels) {
	int test_num = 1000;
	int pos_num = 0;
	int neg_num = 0;
	for (int idx = 0; idx < train_labels.size() && test_labels.size() < test_num; idx++) {
		if (train_labels[idx] == 1 && pos_num < test_num / 2) {
			test_data.push_back(train_data[idx]);
			test_labels.push_back(train_labels[idx]);
			pos_num++;
		}
		else if (train_labels[idx] == -1 && neg_num < test_num / 2) {
			test_data.push_back(train_data[idx]);
			test_labels.push_back(train_labels[idx]);
			neg_num++;
		}
	}
}

int main()
{
	/* Parse mydataset */
	time_t startTime = time(NULL);
	vector<vec_t> train_data;
	vector<label_t> train_labels;
	parse_mydataset_train_data_labels(train_data, train_labels);
	cout << "train_data.size() = " << train_data.size() << endl;
	cout << "train_data[0].size() = " << train_data[0].size() << endl;
	cout << "train_labels.size() = " << train_labels.size() << endl;
	time_t endTime = time(NULL);
	cout << "cost: " << endTime - startTime << " s" << endl;

	/* Generate test data */
	vector<vec_t> test_data;
	vector<label_t> test_labels;
	generate_test_data(train_data, train_labels, test_data, test_labels);

	/* Construct the neural network */
	network<sequential> nn;
	nn << convolutional_layer<tan_h>(64, 64, 5, 1, 3)
	<< average_pooling_layer<tan_h>(60, 60, 3, 2)
	<< convolutional_layer<tan_h>(30, 30, 5, 3, 9)
	<< average_pooling_layer<tan_h>(26, 26, 9, 2)
	<< convolutional_layer<tan_h>(13, 13, 5, 9, 27)
	<< average_pooling_layer<tan_h>(9, 9, 27, 3)
	<< convolutional_layer<tan_h>(3, 3, 3, 27, 91)
	<< fully_connected_layer<tan_h>(91, 200)
	<< fully_connected_layer<tan_h>(200, 200)
	<< fully_connected_layer<tan_h>(200, 2);

	std::cout << "load models..." << std::endl;

	std::cout << "start learning" << std::endl;

	timer t;
	adagrad optimaizer;
	optimaizer.alpha = 0.01f;
	progress_display disp(train_data.size());
	int minibatch_size = 20;
	int epoch = 100;

	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << std::endl << t.elapsed() << "s elapsed." << std::endl;
		tiny_dnn::result res = nn.test(test_data, test_labels);
		std::cout << res.num_success << "/" << res.num_total << std::endl;

		nn.save("LeNet-weights");
		cout << "The network was saved." << endl;

		disp.restart(train_data.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() { disp += minibatch_size; };

	nn.train<mse>(optimaizer, train_data, train_labels, minibatch_size, epoch,
		on_enumerate_minibatch, on_enumerate_epoch);

    return 0;
}

