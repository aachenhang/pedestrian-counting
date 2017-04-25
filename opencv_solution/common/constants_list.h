#pragma once
#ifdef COMMON_EXPORTS
#define COMMONHEADER_API __declspec(dllexport)   
#else  
#define COMMONHEADER_API __declspec(dllimport)   
#endif 


#include <opencv2/core/cvstd.hpp>

using namespace cv;

extern COMMONHEADER_API String positive_samples_file;
extern COMMONHEADER_API String negative_samples_file;
extern COMMONHEADER_API String positive_hard_samples_file;
extern COMMONHEADER_API String negative_hard_samples_file;
extern COMMONHEADER_API String CelebA_dataset_file;
//extern String svm_file;
extern COMMONHEADER_API String svm_file;
extern COMMONHEADER_API String svm_cnn_file;
extern COMMONHEADER_API String image_test_file;
extern COMMONHEADER_API String NN_FILE;
extern COMMONHEADER_API String NN_LAYER_INFROMATION_FILE;
extern COMMONHEADER_API String ALL_IMAGE_FILE;
extern COMMONHEADER_API String TINY_ALL_IMAGE_FILE;
/* convolution neural network model file address */
extern COMMONHEADER_API String CNN_MODEL_TMP;
extern COMMONHEADER_API int const positive_num;
extern COMMONHEADER_API int const negative_num;
extern COMMONHEADER_API int const CelebA_num;
extern COMMONHEADER_API int const positive_hard_num;
extern COMMONHEADER_API int const negative_hard_num;
extern COMMONHEADER_API int const FILL_ZERO;