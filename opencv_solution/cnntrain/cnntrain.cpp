// cnntrain.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;
using namespace std;

static char myWaitKey(int milisecond = 0) {
	cout << "waiting key for " << milisecond/1000 << " seconds" << endl;
	HANDLE stdinHandle = GetStdHandle(STD_INPUT_HANDLE);
	time_t startTime = time(NULL);
	while (time(NULL) < startTime + milisecond / 1000) {
		if (WaitForSingleObject(stdinHandle, 100) == WAIT_OBJECT_0)
		{
			INPUT_RECORD record;
			DWORD numRead;
			if (!ReadConsoleInput(GetStdHandle(STD_INPUT_HANDLE), &record, 1, &numRead)) {
				// hmm handle this error somehow...
				continue;
			}

			if (record.EventType != KEY_EVENT) {
				// don't care about other console events
				continue;
			}

			if (!record.Event.KeyEvent.bKeyDown) {
				// really only care about keydown
				continue;
			}
			char res = record.Event.KeyEvent.uChar.AsciiChar;
			if (res < 'a' || 'z' < res) {
				continue;
			}

			// if you're setup for ASCII, process this:
			//record.Event.KeyEvent.uChar.AsciiChar
			return res;
		}
	}
	return 0;
}

static void parse_mydataset_train_data_labels(vector<vec_t> &train_data,
									vector<label_t> &train_labels) {
	/* Load allData & allLabel from the txt */
	Mat allDataGrey, allLabel;
	FileStorage file(ALL_IMAGE_FILE, FileStorage::READ);
	file["allDataGrey"] >> allDataGrey;
	file["allLabel"] >> allLabel;
	file.release();
	cout << "allData.size() = " << allDataGrey.size() << endl;
	cout << "allLabel.size() = " << allLabel.size() << endl;

	for (int i = 0; i < allDataGrey.rows; i++) {
		vec_t vec(allDataGrey.cols);
		for (int j = 0; j < allDataGrey.cols; j++) {
			vec[j] = static_cast<double>(2.0 * allDataGrey.at<uint8_t>(i, j) / 255 - 1.0);
		}
		train_data.push_back(vec);
		train_labels.push_back(label_t(allLabel.at<int>(i, 0) == 1 ? 1 : 0));

		/* Add mirror data */
		if (allLabel.at<int>(0,i) == 1) {
			vec_t vecMirror(allDataGrey.cols);
			for (int j = 0; j < allDataGrey.cols; j++) {
				vecMirror[j] = static_cast<double>(2.0 * allDataGrey.at<uint8_t>(i, j-j%64*2+63) / 255 - 1.0);
			}
			train_data.push_back(vecMirror);
			train_labels.push_back(label_t(1));
		}
	}

}


static void parse_mydataset_test_data_labels(vector<vec_t> &test_data,
	vector<label_t> &test_labels) {
	/* Load allData & allLabel from the txt */
	Mat allDataGrey, allLabel;
	//FileStorage file(ALL_IMAGE_FILE, FileStorage::READ);
	FileStorage file(TINY_ALL_IMAGE_FILE, FileStorage::READ);
	file["allDataGrey"] >> allDataGrey;
	file["allLabel"] >> allLabel;
	file.release();
	cout << "allData.size() = " << allDataGrey.size() << endl;
	cout << "allLabel.size() = " << allLabel.size() << endl;

	for (int i = 0; i < allDataGrey.rows; i++) {
		vec_t vec(allDataGrey.cols);
		for (int j = 0; j < allDataGrey.cols; j++) {
			vec[j] = static_cast<double>(2.0 * allDataGrey.at<uint8_t>(i, j) / 255 - 1.0);
		}
		test_data.push_back(vec);
		test_labels.push_back(label_t(allLabel.at<int>(i, 0) == 1 ? 1 : 0));

		/* Add mirror data */
		if (allLabel.at<int>(0, i) == 1) {
			vec_t vecMirror(allDataGrey.cols);
			for (int j = 0; j < allDataGrey.cols; j++) {
				vecMirror[j] = static_cast<double>(2.0 * allDataGrey.at<uint8_t>(i, j - j % 64 * 2 + 63) / 255 - 1.0);
			}
			test_data.push_back(vecMirror);
			test_labels.push_back(label_t(1));
		}
	}
}


static void generate_test_data(vector<vec_t> &train_data,
							vector<label_t> &train_labels,
							vector<vec_t> &test_data,
							vector<label_t> &test_labels) {
	int test_num = 1500;
	int pos_num = 0;
	int neg_num = 0;
	for (int idx = 0; idx < train_labels.size() && test_labels.size() < test_num; idx++) {
		if (train_labels[idx] == 1 && pos_num < test_num / 3) {
			test_data.push_back(train_data[idx]);
			test_labels.push_back(train_labels[idx]);
			pos_num++;
		}
		else if (train_labels[idx] == 0 && neg_num < test_num / 3 * 2) {
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
	parse_mydataset_test_data_labels(test_data, test_labels);
	//generate_test_data(train_data, train_labels, test_data, test_labels);
	cout << "test_data.size() = " << test_data.size() << endl;
	cout << "test_data[0].size() = " << test_data[0].size() << endl;
	cout << "test_label.size() = " << test_labels.size() << endl;

	/* Construct the neural network */
	network<sequential> nn;
	nn << convolutional_layer<tan_h>(64, 64, 5, 1, 2)
		<< average_pooling_layer<tan_h>(60, 60, 2, 2)
		<< convolutional_layer<tan_h>(30, 30, 5, 2, 4)
		<< average_pooling_layer<tan_h>(26, 26, 4, 2)
		<< convolutional_layer<tan_h>(13, 13, 5, 4, 8)
		<< fully_connected_layer<tan_h>(648, 600)
		<< fully_connected_layer<tan_h>(600, 600)
		<< fully_connected_layer<tan_h>(600, 2);


	std::cout << "load models..." << std::endl;
	//nn.load(NN_FILE);

	std::cout << "start learning" << std::endl;

	timer t;
	adagrad optimaizer;
	optimaizer.alpha = 0.0001f;
	progress_display disp(train_data.size());
	int minibatch_size = 20;
	int epoch = 2000;
	int epochcnt = 0;

	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << t.elapsed() << "s elapsed." << std::endl;
		tiny_dnn::result res = nn.test(test_data, test_labels);
		std::cout << res.num_success << "/" << res.num_total << std::endl;

		nn.save(NN_FILE);
		cout << "The network was saved at: " << NN_FILE << endl;

		char input = myWaitKey(2000);
		if (input == 'c') {
			string s;
			cout << "Input the alpha" << endl;
			getline(cin, s);
			stringstream stream(s);
			stream >> optimaizer.alpha;
		}
		cout << "optimatizer.apha = " << optimaizer.alpha << endl;
		cout << "epoch = " << epochcnt << endl;
		epochcnt++;

		disp.restart(train_data.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() { disp += minibatch_size; };


	/*nn.train<mse>(optimaizer, test_data, test_labels, minibatch_size, 400,
		on_enumerate_minibatch, on_enumerate_epoch);*/
	

	nn.train<mse>(optimaizer, train_data, train_labels, minibatch_size, epoch,
		on_enumerate_minibatch, on_enumerate_epoch);

    return 0;
}

