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
		case 'g':
			cout << "getsizeofimg" << endl;
			getsizeofimg();
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
			double hitThreshold, finalThreshold;
			cin >> hitThreshold >> finalThreshold;
			hog_svm_detect(hitThreshold, finalThreshold);
			break;
		case 'q':
			return 0;
		default:
			break;
		}

	}
	return 0;
}
