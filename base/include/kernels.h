#pragma once

#include "nppdefs.h"
#include <string>


#define BASIC "basic"
#define M_32 "m_32"
#define M_4K "m_4k"

void launchAddKernel(const Npp8u* src1, const Npp8u* src2, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method);
void launchAddCKernel(const Npp8u* src1, const Npp32u value, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method);
void launchMulCKernel(const Npp8u* src1, const Npp32u value, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method);
void launchAddCMulCKernel(const Npp8u* src, const Npp32u addValue, const Npp32f mulValue, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method);
void launchBrightnessContrast_uv_int8(const Npp8u* src, const Npp32s addValue, const Npp32f mulValue, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method);

void launch_yuv420torgb(const Npp8u* Y, const Npp8u* U, const Npp8u* V, Npp8u* R, Npp8u* G, Npp8u* B, int step_y, int step_uv, NppiSize size, cudaStream_t stream, std::string method);
void launch_rgbtoyuv420(const Npp8u* R, const Npp8u* G, const Npp8u* B, Npp8u* Y, Npp8u* U, Npp8u* V, int step_y, int step_uv, NppiSize size, cudaStream_t stream, std::string method);
void launch_rgbhuesaturation(const Npp8u* r, const Npp8u* g, const Npp8u* b, Npp8u* R, Npp8u* G, Npp8u* B, Npp32f hue, Npp32f saturation, int step, NppiSize size, cudaStream_t stream, std::string method);
void launch_yuv420huesaturation(const Npp8u* y, const Npp8u* u, const Npp8u* v, Npp8u* Y, Npp8u* U, Npp8u* V, Npp32f brighness, Npp32f contrast, Npp32f hue, Npp32f saturation, int step_y, int step_uv, NppiSize size, cudaStream_t stream, std::string method);


void launchUVOverlayKernel(const Npp8u* src1, const Npp8u* src2, Npp8u* dst, Npp32f alpha, int step, NppiSize size, cudaStream_t stream);