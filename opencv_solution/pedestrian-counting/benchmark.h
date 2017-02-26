#pragma once
#include <iostream>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;
#include <mat.h>
#include <time.h>

#include "foreground_extract.h"
#include "test.h"

time_t benchmark_time(int (*func)(char*), char* videofile) {
	time_t start, end;

	start = time(NULL);
	func(videofile);
	end = time(NULL);

	return end - start;
}


int benchmark_foreground_extract(char* videofile) {

	//cout << "The executing time is " << benchmark_time(mog2, videofile) << "s" << endl; //93s
	//cout << "The executing time is " << benchmark_time(Vibe, videofile) << "s" << endl; //264s
	cout << "The executing time is " << benchmark_time(knnextract, videofile) << "s" << endl;	//1014s
	
	return 0;
}
