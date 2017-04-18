#include <iostream>
#include <vector>


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <tiny_dnn\tiny_dnn.h>


using namespace std;
using namespace cv;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

#include "constants_list.h"
#include "process.h"


void getsizeofimg() {

	const char* imgfilename = "F:/Downloads/mydataset/0.jpg";
	Mat img = imread(imgfilename);
	cout << img.cols << endl;	//The result is 1920
}

void test() {
	network<sequential> nn;
	nn.load("LeNet-weights");
	cout << nn.depth() << endl;
	for (int i = 0; i < nn.depth(); i++) {
		vector<tensor_t> output = nn[i]->output();
		cout << "output size is " << output.size() << endl;
		for (tensor_t &tensor : output) {
			cout << "tensor size is" << tensor.size() << endl;
			for (vec_t &vec : tensor) {
				cout << "vec size is" << vec.size() << endl;
				for (float f : vec) {
					cout << f << "<<";
				}
				cout << endl;
			}
		}

	}
}