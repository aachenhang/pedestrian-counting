// image_origin_matrix_to_txt.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace cv;
using namespace std;

static void collect_data(Mat &alldata,
	Mat &alllabels,
	String filepath,
	int label,
	int numofsample,
	int fillflag = 0)
{
	for (int i = 0; i <= numofsample; i++) {
		Mat descriptors, mirrordescriptors;
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
			Mat img = imread(stream.str());
			if (fillflag == 1) {
				int cols = img.cols;
				int rows = img.rows;
				Mat mid = img(Rect(0, (rows - cols) / 2, cols, cols));
				resize(mid, img, Size(64, 64));
			}
			resize(img, descriptors, Size(64 * 64, 1));
			alldata.push_back(descriptors);
			alllabels.push_back(label);
			/* Compute the mirror image *//*
			Mat imgmirror = imMirror(img);
			hog.compute(imgmirror, mirrordescriptors);
			mirrordescriptors.push_back(label);
			alldescriptors.push_back(mirrordescriptors);*/
		}
		else {
			cout << "miss: " << stream.str() << endl;
		}
	}
}

static void collect_data_ingrey(Mat &alldatagrey,
	String filepath,
	int label,
	int numofsample,
	int fillflag = 0)
{
	for (int i = 0; i <= numofsample; i++) {
		Mat descriptors, mirrordescriptors;
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
			Mat img = imread(stream.str(), IMREAD_GRAYSCALE);
			if (fillflag == 1) {
				int cols = img.cols;
				int rows = img.rows;
				Mat mid = img(Rect(0, (rows - cols) / 2, cols, cols));
				resize(mid, img, Size(64, 64));
			}
			resize(img, descriptors, Size(64 * 64, 1));
			alldatagrey.push_back(descriptors);
		}
		else {
			cout << "miss: " << stream.str() << endl;
		}
	}
}


void image_origin_matrix_to_txt() {
	time_t startTime = time(NULL);

	Mat allData, allDataGrey, allLabel;
	
	///* Collect color data and label */
	collect_data(allData, allLabel, positive_samples_file, 1, positive_num);
	collect_data(allData, allLabel, negative_samples_file, -1, negative_num);
	collect_data(allData, allLabel, positive_hard_samples_file, 1, positive_hard_num);
	collect_data(allData, allLabel, negative_hard_samples_file, -1, negative_hard_num);

	/* Collect grey data */
	collect_data_ingrey(allDataGrey, positive_samples_file, 1, positive_num);
	collect_data_ingrey(allDataGrey, negative_samples_file, -1, negative_num);
	collect_data_ingrey(allDataGrey, positive_hard_samples_file, 1, positive_hard_num);
	collect_data_ingrey(allDataGrey, negative_hard_samples_file, -1, negative_hard_num);


	///* Collect color data and label */
	//collect_data(allData, allLabel, positive_samples_file, 1, 10);
	//collect_data(allData, allLabel, negative_samples_file, -1, 10);
	//collect_data(allData, allLabel, positive_hard_samples_file, 1, 10);
	//collect_data(allData, allLabel, negative_hard_samples_file, -1, 10);

	///* Collect grey data */
	//collect_data_ingrey(allDataGrey, positive_samples_file, 1, 10);
	//collect_data_ingrey(allDataGrey, negative_samples_file, -1, 10);
	//collect_data_ingrey(allDataGrey, positive_hard_samples_file, 1, 10);
	//collect_data_ingrey(allDataGrey, negative_hard_samples_file, -1, 10);
	

	/* Print the size of data */
	//cout << "allData.size() = " << allData.size() << endl;
	cout << "allDataGrey.size() = " << allDataGrey.size() << endl;
	cout << "allLabel.size() = " << allLabel.size() << endl;

	/* Write the data */
	FileStorage file(ALL_IMAGE_FILE, FileStorage::WRITE);
	//file << "allData" << allData;
	file << "allDataGrey" << allDataGrey;
	file << "allLabel" << allLabel;

	/* Print cost time */
	time_t endTime = time(NULL);
	cout << "cost " << endTime - startTime << " s" << endl;
}

int main()
{
	image_origin_matrix_to_txt();
	return 0;
}

