// cnnlayer2svm.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;
using namespace cv;
using namespace ml;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;


static int test_file_num = 10;


void svm_cnn_save() {
	/* Initial the time counter */
	time_t start = time(NULL);


	/* Initial the SVM */
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setC(0.01);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));

	/* Load featureMat and labelMat */
	FileStorage file(NN_LAYER_INFROMATION_FILE, FileStorage::READ);
	Mat featureMat;
	Mat labelMat;
	file["featureMat"] >> featureMat;
	file["labelMat"] >> labelMat;

	/* Print the Size of featureMat and labelMat */
	cout << featureMat.size() << endl;
	cout << labelMat.size() << endl;

	/* Print the picture reading time */
	time_t end = time(NULL);
	cout << "cost: " << end - start << "s" << endl;

	/* Train the svm */
	cout << "Trainning starting..." << endl;
	start = time(NULL);
	cout << "The varCount is " << svm->getVarCount() << endl;
	svm->train(featureMat, ROW_SAMPLE, labelMat);
	svm->save(svm_cnn_file);
	end = time(NULL);
	cout << "cost: " << end - start << "s" << endl;
	cout << "save: " << svm_cnn_file << endl;
}


int main()
{
	svm_cnn_save();
    return 0;
}

