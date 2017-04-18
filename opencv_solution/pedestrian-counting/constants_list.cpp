#include <opencv2\core.hpp>

using namespace cv;

String positive_samples_file = "F:/Downloads/mydataset/positive_sample/";
String negative_samples_file = "F:/Downloads/mydataset/negative_sample/";
String positive_hard_samples_file = "F:/Downloads/mydataset/positive_hard_sample/";
String negative_hard_samples_file = "F:/Downloads/mydataset/negative_hard_sample/";
String CelebA_dataset_file = "E:/BaiduNetdiskDownload/CelebA/Img/img_align_celeba/";
String svm_file = "F:/Downloads/mydataset/svm_3000_25000.xml";
String svm_cnn_file = "F:/Downloads/mydataset/svm_cnn.xml";
String image_test_file = "F:/Downloads/mydataset/34.jpg";
extern int const positive_num = 2020;
extern int const negative_num = 25000;
extern int const CelebA_num = 202599;
extern int const positive_hard_num = 1000;
extern int const negative_hard_num = 100;
extern int const FILL_ZERO = 1;
//size of hog  = 1765
//current alpha = 0.01
//String svm_file = "F:/Downloads/mydataset/svm.xml";