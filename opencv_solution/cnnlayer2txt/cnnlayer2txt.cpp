// cnnlayer2txt.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;
using namespace cv;
using namespace ml;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

static void computeDescriptor(vector<vector<float>> &alldescriptors,
	String filepath,
	int label,
	int numofsample,
	network<sequential> &nn,
	int fillflag = 0);

static void computeDescriptor(vector<vector<float>> &alldescriptors,
	String filepath,
	int label,
	int numofsample,
	network<sequential> &nn,
	int fillflag)
{
	for (int i = 0; i <= numofsample; i++) {
		vector<float> descriptors, mirrordescriptors;
		stringstream stream;
		if (fillflag == 1) {
			stream << filepath << setfill('0') << setw(6) << i << ".jpg";
		}
		else {
			stream << filepath << i << ".jpg";
		}
		ifstream f(stream.str());
		if (f.good()) {
			cout << "compute " << stream.str() << endl;
			Mat imgGrey = imread(stream.str(), IMREAD_GRAYSCALE);
			cv::Mat_<uint8_t> resized;
			if (fillflag == 1) {
				int cols = imgGrey.cols;
				int rows = imgGrey.rows;
				Mat mid = imgGrey(Rect(0, (rows - cols) / 2, cols, cols));
				resize(mid, imgGrey, Size(64, 64));
			}
			vec_t d;
			cv::resize(imgGrey, resized, cv::Size(64, 64));
			std::transform(resized.begin(), resized.end(), std::back_inserter(d),
				[=](uint8_t c) { return c * (1.0f - (-1.0f)) / 255.0 + (-1.0f); });
			nn.predict(d);

			vec_t vec = nn[nn.depth() - 2]->output().front().front();
			//cout << "vec.size() = " << vec.size() << endl;
			for (float f : vec) {
				descriptors.push_back(f);
			}

			/*cout << "This is the float in the descriptors" << endl;
			for (float f : descriptors) {
			cout << f << " ";
			}
			cout << endl;*/

			descriptors.push_back(label);
			alldescriptors.push_back(descriptors);

			///* Compute the mirror image */
			//Mat imgmirror = imMirror(imgGrey);
			//for (float f : nn[nn.depth() - 2]->output().front().front()) {
			//	mirrordescriptors.push_back(f);
			//}
			//mirrordescriptors.push_back(label);
			//alldescriptors.push_back(mirrordescriptors);
		}
		else {
			cout << "miss: " << stream.str() << endl;
		}
	}

}


void save_cnn_layer_information() {
	/* Initial the time counter */
	time_t start = time(NULL);

	/* Initial the CNN */
	network<sequential> nn;
	nn.load(NN_FILE);

	/* Initial the descriptors */
	vector<vector<float>> alldescriptors;

	/* Compute the positive samples from my dataset */
	computeDescriptor(alldescriptors, positive_samples_file, 1, positive_num, nn);
	cout << "descriptors number: " << alldescriptors.size() << endl;


	/* Compute the positive samples from CelebA dataset *//*
	computeDescriptor(alldescriptors, CelebA_dataset_file, 1, CelebA_num, hog, FILL_ZERO);
	cout << "descriptors number: " << alldescriptors.size() << endl;*/

	/* Compute the negative samples */
	computeDescriptor(alldescriptors, negative_samples_file, -1, negative_num, nn);
	cout << "descriptors number: " << alldescriptors.size() << endl;

	/* Compute the positive hard samples */
	computeDescriptor(alldescriptors, positive_hard_samples_file, 1, positive_hard_num, nn);
	cout << "descriptors number: " << alldescriptors.size() << endl;

	/* Compute the negative hard samples */
	computeDescriptor(alldescriptors, negative_hard_samples_file, -1, negative_hard_num, nn);
	cout << "descriptors number: " << alldescriptors.size() << endl;


	/* Set the featureMat and labelMat */
	int rows = alldescriptors.size();
	int cols = alldescriptors.begin()->size() - 1;
	Mat featureMat(rows, cols, CV_32FC1);
	Mat labelMat(rows, 1, CV_32SC1);
	for (int i = 0; i < rows; i++) {
		vector<float> descriptor = alldescriptors[i];
		for (int j = 0; j < descriptor.size() - 1; j++) {
			featureMat.at<float>(i, j) = descriptor[j];
		}
		labelMat.at<int>(i, 0) = descriptor[descriptor.size() - 1];	//Notice: Here the float is converted to int
	}

	/* Storge featureMat and labelMat */
	FileStorage file(NN_LAYER_INFROMATION_FILE, FileStorage::WRITE);
	file << "featureMat" << featureMat;
	file << "labelMat" << labelMat;

	/* Print the picture reading time */
	time_t end = time(NULL);
	cout << "cost: " << end - start << "s" << endl;
}


int main()
{
	save_cnn_layer_information();
    return 0;
}

