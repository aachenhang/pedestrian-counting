#include <iostream>
#include <stdio.h>

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
#include "test.h"

int main(int argc, char** argv) {
	//char videofile[] = "F:/Downloads/8116_IP_01_20130224173726_20130224180008_2057515_0001.mp4";
	char videofile[] = "F:/Downloads/8116_IP_segment_0.mp4";
	
	plotdemo();
	return 0;
}
