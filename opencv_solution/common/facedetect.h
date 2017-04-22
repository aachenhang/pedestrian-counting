#pragma once

#ifdef COMMON_EXPORTS  
#define COMMONHEADER_API __declspec(dllexport)   
#else  
#define COMMONHEADER_API __declspec(dllimport)   
#endif 

#include "opencv2\core\types.hpp"
#include <vector>
using namespace std;
using namespace cv;

static void help();


COMMONHEADER_API vector<Rect> facedetect_main(int argc = 0, char** argv = NULL);