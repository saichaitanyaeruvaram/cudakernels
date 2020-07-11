#pragma once

#include <thread>
#include <cuda_runtime_api.h>
#include <iostream>

// tests
void testAdd();
void testAddNPP();













inline bool check(cudaError_t e, int iLine, const char *szFile) {
	if (e != cudaSuccess) {
		const char *szErrName = cudaGetErrorString(e);
		std::cout << "CUDA driver API error " << szErrName << " at line " << iLine << " in file " << szFile << std::endl;
		return false;
	}
	return true;
}

#define ck(call) check(call, __LINE__, __FILE__)


void getDeviceBuffer(int width, int height, int value, void*& buffer, int& step);

void profile(std::function<void()> compute);

