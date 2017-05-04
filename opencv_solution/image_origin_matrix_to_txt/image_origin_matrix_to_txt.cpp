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

	Mat allData, allDataGrey, allLabel, allDataBinary;
	
	///* Collect color data and label */
	//collect_data(allData, allLabel, positive_samples_file, 1, 7000);
	//collect_data(allData, allLabel, negative_samples_file, -1, 7000);
	//collect_data(allData, allLabel, positive_hard_samples_file, 1, positive_hard_num);
	//collect_data(allData, allLabel, negative_hard_samples_file, -1, negative_hard_num);

	///* Collect grey data */
	//collect_data_ingrey(allDataGrey, positive_samples_file, 1, 7000);
	//collect_data_ingrey(allDataGrey, negative_samples_file, -1, 7000);
	//collect_data_ingrey(allDataGrey, positive_hard_samples_file, 1, positive_hard_num);
	//collect_data_ingrey(allDataGrey, negative_hard_samples_file, -1, negative_hard_num);


	/* Collect color data and label */
	collect_data(allData, allLabel, positive_samples_file, 1, 500);
	collect_data(allData, allLabel, negative_samples_file, -1, 500);
	collect_data(allData, allLabel, positive_hard_samples_file, 1, 100);
	collect_data(allData, allLabel, negative_hard_samples_file, -1, 100);

	/* Collect grey data */
	collect_data_ingrey(allDataGrey, positive_samples_file, 1, 500);
	collect_data_ingrey(allDataGrey, negative_samples_file, -1, 500);
	collect_data_ingrey(allDataGrey, positive_hard_samples_file, 1, 100);
	collect_data_ingrey(allDataGrey, negative_hard_samples_file, -1, 100);

	/* binaryzation */
	for (int i = 0; i < allDataGrey.rows; i++) {
		int ave = 0;
		for (int j = 0; j < allDataGrey.cols; j++) {
			ave += allDataGrey.at<uint8_t>(i, j);
		}
		ave /= (64 * 64);
		Mat binary(1, 64*64, CV_8UC1);
		for (int j = 0; j < allDataGrey.cols; j++) {
			binary.at<uint8_t>(0, j) = allDataGrey.at<uint8_t>(i, j) > ave ? 255 : 0;
		}
		allDataBinary.push_back(binary);
	}
	

	/* Print the size of data */
	cout << "allData.size() = " << allData.size() << endl;
	cout << "allDataGrey.size() = " << allDataGrey.size() << endl;
	cout << "allLabel.size() = " << allLabel.size() << endl;
	cout << "allDataBinary.size() = " << allDataBinary.size() << endl;

	/* Write the data */
	FileStorage file(TINY_ALL_IMAGE_FILE, FileStorage::WRITE);
	file << "allData" << allData;
	file << "allDataGrey" << allDataGrey;
	file << "allLabel" << allLabel;
	file << "allDataBinary" << allDataBinary;

	/* Print cost time */
	time_t endTime = time(NULL);
	cout << "cost " << endTime - startTime << " s" << endl;
}


static void image_origin_matrix_to_tsv() {
	Mat allDataGrey, allLabel;
	FileStorage file(ALL_IMAGE_FILE, FileStorage::READ);
	file["allDataGrey"] >> allDataGrey;
	file["allLabel"] >> allLabel;
	cout << "allDataGrey.size() = " << allDataGrey.size() << endl;
	cout << "allLabel.size() = " << allLabel.size() << endl;
	file.release();

	/*fstream datafile("H:/Pro/visualize/data.tsv");
	for (int i = 0; i < allDataGrey.rows; i++) {
		for (int j = 0; j < allDataGrey.cols - 1; j++) {
			datafile << static_cast<int>(allDataGrey.at<uint8_t>(i, j)) << "\t";
		}
		datafile << static_cast<int>(allDataGrey.at<uint8_t>(i, allDataGrey.cols - 1)) << endl;
	}
	datafile.close();*/

	fstream labelfile("H:/Pro/visualize/label.tsv");
	for (int i = 0; i < allLabel.rows; i++) {
		labelfile << (allLabel.at<int>(i, 0) == 1 ? 1 : 0) << endl;
	}

}


static void show_matrix_label_distribute() {
	Mat allDataGrey, allLabel;
	FileStorage file(TINY_ALL_IMAGE_FILE, FileStorage::READ);
	file["allDataGrey"] >> allDataGrey;
	file["allLabel"] >> allLabel;
	cout << "allDataGrey.size() = " << allDataGrey.size() << endl;
	cout << "allLabel.size() = " << allLabel.size() << endl;
	file.release();

	int poscnt = 0, negcnt = 0;
	for (int i = 0; i < allLabel.rows; i++) {
	if (allLabel.at<int>(i, 0) == 1) {
	poscnt++;
	}
	else {
	negcnt++;
	}
	}
	cout << "poscnt = " << poscnt << endl;
	cout << "negcnt = " << negcnt << endl;
}


int main()
{
	image_origin_matrix_to_txt();
	//image_origin_matrix_to_tsv();
	//show_matrix_label_distribute();
	return 0;
}

