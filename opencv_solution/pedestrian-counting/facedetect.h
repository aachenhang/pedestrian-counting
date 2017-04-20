#pragma once

#include "opencv2\core\types.hpp"
#include <vector>
using namespace std;
using namespace cv;

static void help();


vector<Rect> facedetect_main(int argc = 0, char** argv = NULL);