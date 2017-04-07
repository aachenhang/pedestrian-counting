#pragma once

#include <vector>
#include <iostream>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;


vector<Rect> mergeLocation(const vector<Rect>& foundLocations);
int findRoot(int pos);
void merge(int a, int b);
bool canMerge(Rect a, Rect b);
bool cmp(Rect a, Rect b);
