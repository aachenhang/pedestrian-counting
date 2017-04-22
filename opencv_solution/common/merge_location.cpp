#pragma once
#ifdef COMMON_EXPORTS  
#define COMMONHEADER_API __declspec(dllexport)   
#else  
#define COMMONHEADER_API __declspec(dllimport)   
#endif 
#include <vector>
#include <iostream>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;


int mergeset[10005];

COMMONHEADER_API int findRoot(int pos) {
	return mergeset[pos] == -1 ? pos : mergeset[pos] = findRoot(mergeset[pos]);
}

COMMONHEADER_API void merge(int a, int b) {
	if (a > b)	swap(a, b);
	mergeset[b] = findRoot(a);
}


COMMONHEADER_API bool canMerge(Rect a, Rect b) {
	/* step size = 8, can be merge if not more than 2 steps */
	return abs(a.x - b.x) + abs(a.y - b.y) <= 8 * 2;
}


COMMONHEADER_API bool cmp(Rect a, Rect b) {
	return a.x < b.x || a.x == b.x && a.y < b.y;
}

COMMONHEADER_API vector<Rect> mergeLocation(const vector<Rect>& foundLocations) {
	if (foundLocations.size() >= 10005)
		cerr << "foundLocations.size() = " << foundLocations.size() << endl;
	for (int i = 0; i < foundLocations.size(); i++) {
		mergeset[i] = -1;
		for (int j = 0; j < i; j++) {
			if (canMerge(foundLocations[i], foundLocations[j])) {
				merge(i, j);
			}
		}
	}
	vector<Rect> res;
	for (int i = 0; i < foundLocations.size(); i++) {
		int cnt = 0;
		Rect cur(0, 0, 64, 64);
		for (int j = 0; j < foundLocations.size(); j++) {
			if (findRoot(j) == i) {
				cnt++;
				cur.x += foundLocations[j].x;
				cur.y += foundLocations[j].y;
			}
		}
		if (cnt != 0) {
			cur.x /= cnt;
			cur.y /= cnt;
			res.push_back(cur);
		}
	}
	return res;
}



