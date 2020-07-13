#pragma once

#include "nppdefs.h"
#include <string>


#define ADD_BASIC "add_basic"
#define ADD_32 "add_32"

void launchAddKernel(const Npp8u* src1, const Npp8u* src2, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method);