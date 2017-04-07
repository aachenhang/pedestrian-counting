#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
using namespace std;


#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;


#include "process.h"
#include "constants_list.h"

void mouseClick(int event, int x, int y, int flags, void* userdata);

int makeAnnotation(int inputnum);

int catchimage(int inputnum);

int cropimage(int inputnum);

int createNegativeSample(int inputnum, int sum);

void createHardSample();