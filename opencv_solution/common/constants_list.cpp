#ifdef COMMON_EXPORTS
#define COMMONHEADER_API __declspec(dllexport)   
#else  
#define COMMONHEADER_API __declspec(dllimport)   
#endif 

#include <opencv2\core.hpp>


using namespace cv;

COMMONHEADER_API String positive_samples_file = "F:/Downloads/mydataset/positive_sample/";
COMMONHEADER_API String negative_samples_file = "F:/Downloads/mydataset/negative_sample/";
COMMONHEADER_API String positive_hard_samples_file = "F:/Downloads/mydataset/positive_hard_sample/";
COMMONHEADER_API String negative_hard_samples_file = "F:/Downloads/mydataset/negative_hard_sample/";
COMMONHEADER_API String CelebA_dataset_file = "E:/BaiduNetdiskDownload/CelebA/Img/img_align_celeba/";
COMMONHEADER_API String svm_file = "F:/Downloads/mydataset/svm_3000_25000.xml";
COMMONHEADER_API String svm_cnn_file = "F:/Downloads/mydataset/svm_cnn.xml";
COMMONHEADER_API String image_test_file = "F:/Downloads/mydataset/34.jpg";
extern COMMONHEADER_API String NN_FILE = "H:/Pro/NN_Model/LeNet-weights";
extern COMMONHEADER_API String NN_LAYER_INFROMATION_FILE = "H:/Pro/NN_Model/NN_LAYER.txt";
extern COMMONHEADER_API int const positive_num = 2020;
extern COMMONHEADER_API int const negative_num = 25000;
extern COMMONHEADER_API int const CelebA_num = 202599;
extern COMMONHEADER_API int const positive_hard_num = 1000;
extern COMMONHEADER_API int const negative_hard_num = 100;
extern COMMONHEADER_API int const FILL_ZERO = 1;
//size of hog  = 1765
//current alpha = 0.01
//String svm_file = "F:/Downloads/mydataset/svm.xml";