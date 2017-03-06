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
#include "create_dataset.h"
#include "test.h"


int main(int argc, char** argv) {

	makeAnnotation();

	return 0;
}
