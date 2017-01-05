/**
* @file videocapture_starter.cpp
* @brief A starter sample for using OpenCV VideoCapture with capture devices, video files or image sequences
* easy as CV_PI right?
*
*  Created on: Nov 23, 2010
*      Author: Ethan Rublee
*
*  Modified on: April 17, 2013
*      Author: Kevin Hughes
*/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
#include <mat.h>
#include "mxAnalyze.h"
int diagnose(const char *file) {
	MATFile *pmat;
	const char **dir;
	const char *name;
	int	  ndir;
	int	  i;
	mxArray *pa;

	printf("Reading file %s...\n\n", file);

	/*
	* Open file to get directory
	*/
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("Error opening file %s\n", file);
		return(1);
	}

	/*
	* get directory of MAT-file
	*/
	dir = (const char **)matGetDir(pmat, &ndir);
	if (dir == NULL) {
		printf("Error reading directory of file %s\n", file);
		return(1);
	}
	else {
		printf("Directory of %s:\n", file);
		for (i = 0; i < ndir; i++)
			printf("%s\n", dir[i]);
	}
	mxFree(dir);

	/* In order to use matGetNextXXX correctly, reopen file to read in headers. */
	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n", file);
		return(1);
	}
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("Error reopening file %s\n", file);
		return(1);
	}

	/* Get headers of all variables */
	printf("\nExamining the header for each variable:\n");
	for (i = 0; i < ndir; i++) {
		pa = matGetNextVariableInfo(pmat, &name);
		if (pa == NULL) {
			printf("Error reading in file %s\n", file);
			return(1);
		}
		/* Diagnose header pa */
		printf("According to its header, array %s has %d dimensions\n",
			name, mxGetNumberOfDimensions(pa));
		if (mxIsFromGlobalWS(pa))
			printf("  and was a global variable when saved\n");
		else
			printf("  and was a local variable when saved\n");
		mxDestroyArray(pa);
	}

	/* Reopen file to read in actual arrays. */
	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n", file);
		return(1);
	}
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("Error reopening file %s\n", file);
		return(1);
	}

	/* Read in each array. */
	printf("\nReading in the actual array contents:\n");
	for (i = 0; i<ndir; i++) {
		pa = matGetNextVariable(pmat, &name);
		if (pa == NULL) {
			printf("Error reading in file %s\n", file);
			return(1);
		}
		/*
		* Diagnose array pa
		*/
		printf("According to its contents, array %s has %d dimensions\n",
			name, mxGetNumberOfDimensions(pa));
		if (mxIsFromGlobalWS(pa))
			printf("  and was a global variable when saved\n");
		else
			printf("  and was a local variable when saved\n");

		if (mxIsCell(pa)) {
			cout << "is cell" << endl;
		}
		else if (mxIsStruct(pa)) {
			cout << "is struct" << endl;
		}
		else if (mxIsNumeric(pa)) {
			cout << "is numeric" << endl;
		}
		else {
			cout << "is others" << endl;
		}

		analyze_full(pa);

		mxDestroyArray(pa);
	}


	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n", file);
		return(1);
	}
	printf("Done\n");
	return(0);
}
//hide the local functions in an anon namespace
namespace {
	int process(VideoCapture& capture) {
		int n = 0;
		char filename[200];
		Mat frame;
		
		/*capture >> frame;
		cout << "size: " << frame.size() << endl;
		cout << "channels: " << frame.channels() << endl;
		cout << "depth: " << frame.depth() << endl;
		cout << "elemSize: " << frame.elemSize() << endl;
		cout << "elemSize1: " << frame.elemSize1() << endl;
		cout << "total: " << frame.total() << endl;
		cout << "type: " << frame.type() << endl;
		cout << frame.row(0).size() << endl;
		cout << frame.row(0) << endl;
*/
		int i = 0;
		double t = (double)getTickCount();
		//for (;;) {
		//	capture >> frame;
		//	printf("%d\n", i++);
		//	if (frame.empty())
		//		break;

			//imshow(window_name, frame);
			//char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

			//switch (key) {
			//case 'q':
			//case 'Q':
			//case 27: //escape key
			//	return 0;
			//case ' ': //Save an image
			//	sprintf(filename, "filename%.3d.jpg", n++);
			//	imwrite(filename, frame);
			//	cout << "Saved " << filename << endl;
			//	break;
			//default:
			//	cout << "Default " << filename << endl;
			//	break;
			//}
		//}

		//t = ((double)getTickCount() - t) / getTickFrequency();
		//cout << "Times passed in seconds: " << t << endl;//265.83s-33800frames
		return 0;
	}
}

int main(int argc, char** argv) {
	//cv::CommandLineParser parser(ac, av, "{help h||}{@input||}");
	//if (parser.has("help"))
	//{
	//	return 0;
	//}
	////std::string arg = parser.get<std::string>("@input");
	//std::string arg = "F:/Downloads/8116_IP_01_20130224173726_20130224180008_2057515_0001.mp4";

	//if (arg.empty()) {
	//	return 1;
	//}
	//VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file or image sequence
	//if (!capture.isOpened()) {
	//	cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
	//	return 1;
	//}
	//return process(capture);
	int result;

	char matfile[] = "F:/Downloads/Compressed/mall_dataset/mall_feat.mat";
	argc = 2;
	if (argc > 1)
		result = diagnose(matfile);
	else {
		result = 0;
		printf("Usage: matdgns <matfile>");
		printf(" where <matfile> is the name of the MAT-file");
		printf(" to be diagnosed\n");
	}

	return (result == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
