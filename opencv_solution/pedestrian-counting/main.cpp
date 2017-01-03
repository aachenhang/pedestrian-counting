#include <iostream>
#include <opencv2\core\core.hpp>

using namespace cv;
using namespace std;

int main(int argc, int** argv){
	Mat a(2, 2, CV_8UC3, Scalar(0, 0, 225));
	cout << a << endl;
	return 0;
}