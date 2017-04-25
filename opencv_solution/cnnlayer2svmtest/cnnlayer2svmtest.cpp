// cnnlayer2svmtest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;
using namespace cv;
using namespace ml;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

static void hog_svm_load(HOGDescriptor &hog, String file = svm_file);
static void hog_svm_load(HOGDescriptor &hog, String file) {

	/* Load the svm */
	Ptr<SVM> svm = SVM::load(file);
	cout << "load: " << file << endl;

	/* Get alpha and rho */
	Mat supportVector = svm->getSupportVectors();
	int alpharows = supportVector.rows;
	int alphacols = svm->getVarCount();
	Mat alpha(alpharows, alphacols, CV_32F);
	Mat svindex(1, alpharows, CV_64F);
	double rho = svm->getDecisionFunction(0, alpha, svindex);
	alpha.convertTo(alpha, CV_32F);

	/* Get the result Mat */
	Mat result = -1 * alpha * supportVector;

	/* Get the detector vector */
	vector<float> detector;
	CV_Assert(result.cols > 100);
	for (int i = 0; i < result.cols; i++) {
		detector.push_back(result.at<float>(0, i));
	}

	/* Load the hog detector */
	hog.setSVMDetector(detector);
	cout << "load: HOGDescriptor" << endl;
}

void svm_cnn_detect() {
	/* Load the hog detector */
	Size winsize(64, 64), blocksize(16, 16), blockstep(8, 8), cellsize(8, 8);
	int nbins = 9;
	HOGDescriptor hog(winsize, blocksize, blockstep, cellsize, nbins);
	hog_svm_load(hog);


	/* Load the svm_cnn */
	Ptr<SVM> svm_cnn = SVM::load(svm_cnn_file);
	cout << "load: " << svm_cnn_file << endl;

	cout << "cout the feature of sum_cnn: " << endl;
	cout << "varcount is : " << svm_cnn->getVarCount() << endl;

	/* Load the CNN */
	network<sequential> nn;
	nn.load(NN_FILE);

	while (1) {
		/* Detect the image */
		Mat img = imread(image_test_file);
		Mat imgGrey = imread(image_test_file, IMREAD_GRAYSCALE);
		vector<Rect> foundLocations;
		double hitThreshold = 10.0;
		cin >> hitThreshold;
		if (hitThreshold == 0)	break;
		Size winStride(8, 8);
		Size padding(0, 0);
		double scale = 1.05;
		double finalThreshold = 100.0;
		cin >> finalThreshold;
		bool useMeanshiftGrouping = false;
		hog.detectMultiScale(img, foundLocations, hitThreshold, winStride, padding, finalThreshold, useMeanshiftGrouping);

		/* Merge the locations */
		vector<Rect> mergedLocations = mergeLocation(foundLocations);

		cout << "cin the svm predict threshold " << endl;
		float svmThreshold = 0;
		cin >> svmThreshold;
		vector<Rect> res, notFound;
		for (Rect rect : mergedLocations) {
			Mat tmp = imgGrey(rect);
			cv::Mat_<uint8_t> resized;
			vec_t d;
			cv::resize(tmp, resized, cv::Size(64, 64));
			std::transform(resized.begin(), resized.end(), std::back_inserter(d),
				[=](uint8_t c) { return c * (1.0f - (-1.0f)) / 255.0 + (-1.0f); });
			nn.predict(d);
			vec_t vec = nn[nn.depth() - 2]->output().front().front();
			int cols = vec.size();
			Mat featureMat(1, cols, CV_32FC1);
			for (int j = 0; j < cols; j++) {
				float f = vec[j];
				featureMat.at<float>(0, j) = f;
			}
			if (svm_cnn->predict(featureMat) >= svmThreshold) {
				res.push_back(rect);
			} else {
				notFound.push_back(rect);
			}
		}

		/* Print the locations */
		sort(res.begin(), res.end(), cmp);
		for (Rect rect : res) {
			cout << rect.x << " " << rect.y << endl;
		}

		/* Display the rectangle */
		Scalar GREEN = Scalar(0, 255, 0);
		Scalar RED = Scalar(32, 85, 234);
		for (int i = 0; i < res.size(); i++) {
			rectangle(img, res[i], GREEN, 2);
		}
		for (int i = 0; i < notFound.size(); i++) {
			rectangle(img, notFound[i], RED, 3);
		}
		imshow("Result", img);
		waitKey(0);
		imwrite("Result.jpg", img);
		destroyAllWindows();
	}
}

int main()
{
	svm_cnn_detect();
    return 0;
}

