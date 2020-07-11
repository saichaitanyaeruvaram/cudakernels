#pragma once

#include <cuda_runtime_api.h>

inline bool check(cudaError_t e, int iLine, const char *szFile) {
	if (e != cudaSuccess) {
		const char *szErrName = cudaGetErrorString(e);
		std::cout << "CUDA driver API error " << szErrName << " at line " << iLine << " in file " << szFile << std::endl;
		return false;
	}
	return true;
}

#define ck(call) check(call, __LINE__, __FILE__)