// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the TINY_DNN_DLL_GENERATE_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// TINY_DNN_DLL_GENERATE_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef TINY_DNN_DLL_GENERATE_EXPORTS
#define TINY_DNN_DLL_GENERATE_API __declspec(dllexport)
#else
#define TINY_DNN_DLL_GENERATE_API __declspec(dllimport)
#endif

// This class is exported from the tiny_dnn_dll_generate.dll
class TINY_DNN_DLL_GENERATE_API Ctiny_dnn_dll_generate {
public:
	Ctiny_dnn_dll_generate(void);
	// TODO: add your methods here.
};

extern TINY_DNN_DLL_GENERATE_API int ntiny_dnn_dll_generate;

TINY_DNN_DLL_GENERATE_API int fntiny_dnn_dll_generate(void);
