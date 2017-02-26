#ifndef TEST_H
#define TEST_H


#include <cstddef>
#include <ctime>
#include <iostream>


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
//
//#include <libvibe++/ViBe.h>
//#include <libvibe++/distances/Manhattan.h>
#include <stdint.h>

class myclass {
public:
	int add();
};

static __inline void mytest() {
	myclass m;
	printf("%d\n", m.add());
	return;
}


#endif // !TEST_H