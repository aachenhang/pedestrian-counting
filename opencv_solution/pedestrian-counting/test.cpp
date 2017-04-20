#include "stdafx.h"



using namespace std;
using namespace cv;

#include "constants_list.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

void getsizeofimg() {

	const char* imgfilename = "F:/Downloads/mydataset/0.jpg";
	Mat img = imread(imgfilename);
	cout << img.cols << endl;	//The result is 1920
}

void test() {
	network<sequential> nn;
	nn.load("LeNet-weights");

	cv::Mat_<uint8_t> resized;
	vec_t d;
	Mat tmp = imread("F:/Downloads/mydataset/positive_sample/0.jpg", IMREAD_GRAYSCALE);
	cv::resize(tmp, resized, cv::Size(64, 64));
	std::transform(resized.begin(), resized.end(), std::back_inserter(d),
		[=](uint8_t c) { return c * (1.0f - (-1.0f)) / 255.0 + (-1.0f); });

	nn.predict(d);

	for (int layerIdx = 0; layerIdx < nn.depth(); layerIdx++) {
		cout << "This is the layer of " << layerIdx << endl;
		vector<tensor_t> output = nn[layerIdx]->output();
		for (int tensorIdx = 0; tensorIdx < output.size(); tensorIdx++) {
			cout << "This is the tensor of " << tensorIdx << endl;
			/* cout the tensor */
			for (vec_t vec : output[tensorIdx]) {
				for (float f : vec) {
					cout << f << " ";
				}
				cout << endl << endl;
			}
		}
	}

}