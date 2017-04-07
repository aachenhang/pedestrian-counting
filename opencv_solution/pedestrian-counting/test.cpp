#include <iostream>
#include <vector>


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>


using namespace std;
using namespace cv;

#include "constants_list.h"
#include "process.h"


void getsizeofimg() {

	const char* imgfilename = "F:/Downloads/mydataset/0.jpg";
	Mat img = imread(imgfilename);
	cout << img.cols << endl;	//The result is 1920
}

void test() {
	cout << "This is the gatedoor of testing " << endl;
	vector<vector<float>> alldescriptors;

	/* Initial the HOGDescriptor */
	Size winsize(64, 64), blocksize(16, 16), blockstep(8, 8), cellsize(8, 8);
	int nbins = 9;
	HOGDescriptor hog(winsize, blocksize, blockstep, cellsize, nbins);
	
	computeDescriptor(alldescriptors,
		positive_samples_file,
		1,
		positive_num,
		hog);

	cout << alldescriptors.size() << endl;
	cout << alldescriptors[0].size() << endl;
}