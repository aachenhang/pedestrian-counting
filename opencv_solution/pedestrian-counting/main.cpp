#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;
#include <mat.h>
#include <time.h>

#include "foreground_extract.h"
#include "benchmark.h"
#include "create_dataset.h"
#include "test.h"
#include "process.h"
#include "tiny-dnn.h"

void print_help() {
	cout << "*****************************************************************************" << endl;
	cout << "******   This is aachenhang's opencv project to count the pedestrain   ******" << endl;
	cout << "******   Version: 0.1.1        Operation System: win10 64bits          ******" << endl;
	cout << "******   opencv: 3.2           Visual Studio: vs2015/v14               ******" << endl;
	cout << "*****************************************************************************" << endl;
	cout << "Input the order to cal the function: " << endl;
	cout << "c : cropimage(), crop the 64*64 image from the large 1920*1088 image." << endl;
	cout << "m : makeAnnotation(), open the 1920*1088 image to add annotation manually." << endl;
	cout << "i : catchimage(), catch 5 pieces of 1920*1088 images from 1 min's video." << endl;
	cout << "n : createNegativeSample(), create negative sample randomly according to the annotation txt" << endl;
	cout << "N : cal createNegativeSample() for 100 times for all the 20 pieces of 1920*1088 image" << endl;
	cout << "s : hog_svm_save(), cal the svm + hog training" << endl;
	cout << "d : hog_svm_detect(), cal the svm + hog testing" << endl;
	cout << "H : createHardSample(), find wrong answer from positive_sample and negative_sample using svm model" << endl;
	cout << "v : sample1_convnet(), cal the CNN training" << endl;
	cout << "t : convnet_test(), cal the cnn testing" << endl;
	cout << "h : print_help()" << endl;
	cout << "q : quit" << endl;
}

int main(int argc, char** argv) {
	while (1) {
		cout << "/*******select the function:*******/" << endl;
		char c;
		int inputnum;
		int sum;
		cin >> c;
		switch (c)
		{
		case 'c':
			cout << "cropimage" << endl;
			cout << "input the offset: " << endl;
			cin >> inputnum;
			cropimage(inputnum);
			break;
		case 'm':
			cout << "makeAnnotation" << endl;
			cout << "input the offset: " << endl;
			cin >> inputnum;
			makeAnnotation(inputnum);
			break;
		case 'i':
			cout << "catchimage" << endl;
			cout << "input the offset: " << endl;
			cin >> inputnum;
			catchimage(inputnum);
			break;
		case 'n':
			cout << "create negative samples" << endl;
			cout << "input the offset: " << endl;
			cin >> inputnum;
			cout << "input the sum: " << endl;
			cin >> sum;
			createNegativeSample(inputnum, sum);
			break;
		case 'N':
			for (int i = 0; i < 21; i++) {
				createNegativeSample(i, 100);
			}
			break;
		case 's':
			hog_svm_save();
			break;
		case 'd':
			hog_svm_detect();
			break;
		case 'H':
			createHardSample();
			break;
		case 'v':
			sample1_convnet();
			break;
		case 't':
			convnet_test();
			break;
		case 'h':
			print_help();
			break;
		case 'q':
			return 0;
		default:
			break;
		}

	}
	return 0;
}
