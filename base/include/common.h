#pragma once

#include <cuda_runtime_api.h>
#include "npp.h"

inline bool check(cudaError_t e, int iLine, const char *szFile) {
	if (e != cudaSuccess) {
		const char *szErrName = cudaGetErrorString(e);
		std::cout << "CUDA driver API error " << szErrName << " at line " << iLine << " in file " << szFile << std::endl;
		return false;
	}
	return true;
}

#define ck(call) check(call, __LINE__, __FILE__)


inline bool checkNPPStatus(NppStatus e, int iLine, const char *szFile) {
	if (e != NPP_SUCCESS) {
		std::cout << "NPP API error " << e << " at line " << iLine << " in file " << szFile << std::endl;
		return false;
	}
	return true;
}

#define check_nppstatus(call) checkNPPStatus(call, __LINE__, __FILE__)