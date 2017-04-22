#pragma once

#ifdef COMMON_EXPORTS  
#define COMMONHEADER_API __declspec(dllexport)   
#else  
#define COMMONHEADER_API __declspec(dllimport)   
#endif 

#include <vector>
#include "opencv2\core\types.hpp"

using namespace std;
using namespace cv;


COMMONHEADER_API vector<Rect> mergeLocation(const vector<Rect>& foundLocations);
COMMONHEADER_API int findRoot(int pos);
COMMONHEADER_API void merge(int a, int b);
COMMONHEADER_API bool canMerge(Rect a, Rect b);
COMMONHEADER_API bool cmp(Rect a, Rect b);
