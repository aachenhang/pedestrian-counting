// tiny_dnn_dll_generate.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "tiny_dnn_dll_generate.h"


// This is an example of an exported variable
TINY_DNN_DLL_GENERATE_API int ntiny_dnn_dll_generate=0;

// This is an example of an exported function.
TINY_DNN_DLL_GENERATE_API int fntiny_dnn_dll_generate(void)
{
    return 42;
}

// This is the constructor of a class that has been exported.
// see tiny_dnn_dll_generate.h for the class definition
Ctiny_dnn_dll_generate::Ctiny_dnn_dll_generate()
{
    return;
}

#include "tiny_dnn\tiny_dnn.h"
