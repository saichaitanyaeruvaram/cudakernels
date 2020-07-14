#pragma once

#include "nppdefs.h"
#include <string>


#define BASIC "basic"
#define M_32 "m_32"
#define M_4K "m_4k"

void launchAddKernel(const Npp8u* src1, const Npp8u* src2, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method);
void launchAddCKernel(const Npp8u* src1, const Npp32u value, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method);
void launchMulCKernel(const Npp8u* src1, const Npp32u value, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method);
void launchAddCMulCKernel(const Npp8u* src, const Npp32u addValue, const Npp32u mulValue, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method);