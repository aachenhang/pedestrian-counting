// cnninference.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;
using namespace std;


void cnninference() {
	network<sequential> nn;
	nn.load(NN_FILE);

	Mat allDataGrey, allLabel;
	FileStorage file(ALL_IMAGE_FILE, FileStorage::READ);
	file["allDataGrey"] >> allDataGrey;
	file["allLabel"] >> allLabel;
	cout << "allDataGrey.size() = " << allDataGrey.size() << endl;
	cout << "allLabel.size()" << allLabel.size() << endl;


}


int main()
{
	cnninference();
    return 0;
}

