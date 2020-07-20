#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernels.h"
#include <iostream>

__device__ uint8_t
mul_uchar(uint8_t value1, int value2)
{
	uint32_t out = value1 * value2;

	if (out > 0xff)
	{
		return 0xff;
	}

	return static_cast<uint8_t>(out);
}

__device__ uint8_t
muladd_uchar(uint8_t value, uint8_t addValue, int mulValue)
{
	uint32_t out = value * mulValue;

	if (out > 0xff)
	{
		return 0xff;
	}

	out = out + addValue;
	if (out > 0xff)
	{
		return 0xff;
	}

	return static_cast<uint8_t>(out);
}


__global__ void add(const Npp8u* src1, const Npp8u* src2, Npp8u* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	dst[offset] = src1[offset] + src2[offset];
	dst[offset] |= -(dst[offset] < src2[offset]);
}

__global__ void addc(const Npp8u* src, const Npp8u value, Npp8u* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	dst[offset] = src[offset] + value;
	dst[offset] |= -(dst[offset] < value);
}

__global__ void mulc(const Npp8u* src, const Npp8u value, Npp8u* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	uint32_t dstValue = src[offset] * value;
	if (dstValue > 255)
	{
		dst[offset] = 255;
	}
	else
	{
		dst[offset] = dstValue;
	}
}

__global__ void addcmulc(const Npp8u* src, const Npp8u addValue, const Npp8u mulValue, Npp8u* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	int dstValue = src[offset] * mulValue;
	dstValue = dstValue + addValue;
	if (dstValue > 255)
	{
		dstValue = 255;
	}
	dst[offset] = dstValue;
}

__global__ void add_32(const Npp8u* src1, const Npp8u* src2, Npp8u* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	x = x << 5;

	for (auto i = 0; i < 32; i++)
	{
		int offset = y * step + x + i;
		dst[offset] = src1[offset] + src2[offset];
	}
}

// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SIMD.html#group__CUDA__MATH__INTRINSIC__SIMD

#define CLAMP_1(x) x < 0 ? 0 : (x > 1 ? 1 : x)
#define CLAMP_255(x) x < 0 ? 0 : (x > 255 ? 255 : x)
#define CLAMP_int8(x) x < -128 ? -128 : (x > 127 ? 127 : x)

__global__ void add_4k_(const Npp32u* src1, const Npp32u* src2, Npp32u* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	dst[offset] = __vaddus4(src1[offset], src2[offset]);
}

__global__ void add_4k(const uchar4* src1, const uchar4* src2, uchar4* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	dst[offset].x = CLAMP_255(src1[offset].x + src2[offset].x);
	dst[offset].y = CLAMP_255(src1[offset].y + src2[offset].y);
	dst[offset].z = CLAMP_255(src1[offset].z + src2[offset].z);
	dst[offset].w = CLAMP_255(src1[offset].w + src2[offset].w);
}

__global__ void addc_4k(const Npp32u* src, const Npp32u value, Npp32u* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	dst[offset] = __vaddus4(src[offset], value);
}

__global__ void mulc_4k(const uchar4* src, const Npp32u value, uchar4* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	dst[offset].x = mul_uchar(src[offset].x, value);
	dst[offset].y = mul_uchar(src[offset].y, value);
	dst[offset].z = mul_uchar(src[offset].z, value);
	dst[offset].w = mul_uchar(src[offset].w, value);
}

__global__ void addcmulc_4k_nottherightway(const uchar4* src, const Npp8u addValue, const Npp8u mulValue, uchar4* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	dst[offset].x = muladd_uchar(src[offset].x, addValue, mulValue);
	dst[offset].y = muladd_uchar(src[offset].y, addValue, mulValue);
	dst[offset].z = muladd_uchar(src[offset].z, addValue, mulValue);
	dst[offset].w = muladd_uchar(src[offset].w, addValue, mulValue);
}

__global__ void addcmulc_4k(const uchar4* src, const Npp8u addValue, const Npp32f mulValue, uchar4* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	auto& srcValue = src[offset];
	auto &dstValue = dst[offset];
	uint32_t temp = 0;
	temp = srcValue.x*mulValue + addValue;
	dstValue.x = temp > 255 ? 255:temp;
	temp = srcValue.y*mulValue + addValue;
	dstValue.y = temp > 255 ? 255 : temp;
	temp = srcValue.z*mulValue + addValue;
	dstValue.z = temp > 255 ? 255 : temp;
	temp = srcValue.w*mulValue + addValue;
	dstValue.w = temp > 255 ? 255 : temp;
}

#define ADDCMULC_INT8_4k_OP( srcValue, dstValue, mulValue )                                                                       \
    do                                                                                                             						   \
    {                                                                                                                                      \
      int32_t temp = (srcValue - 128)*mulValue;																				   \
	  temp = temp > 127 ? 127: temp;																									   \
	  temp = temp < -128 ? 0: temp+128;                                                                                                    \
	  dstValue = temp;																													   \
    } while (0)																									   			


__global__ void brightnesscontrast_uv_int8_4k(const uchar4* src, const Npp32s addValue, const Npp32f mulValue, uchar4* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	auto& srcValue = src[offset];
	auto &dstValue = dst[offset];
	ADDCMULC_INT8_4k_OP(srcValue.x, dstValue.x, mulValue);
	ADDCMULC_INT8_4k_OP(srcValue.y, dstValue.y, mulValue);
	ADDCMULC_INT8_4k_OP(srcValue.z, dstValue.z, mulValue);
	ADDCMULC_INT8_4k_OP(srcValue.w, dstValue.w, mulValue);
}

void launchAddKernel(const Npp8u* src1, const Npp8u* src2, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method)
{
	if (method == BASIC)
	{
		dim3 block(32, 32);
		dim3 grid((size.width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		add << <grid, block, 0, stream >> > (src1, src2, dst, step, size.width, size.height);
	}
	else if (method == M_32)
	{
		auto width = size.width >> 5;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		add_32 << <grid, block, 0, stream >> > (src1, src2, dst, step, width, size.height);
	}
	else if (method == M_4K)
	{
		auto width = size.width >> 2;
		step = step >> 2;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);

		add_4k << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(src1), reinterpret_cast<const uchar4*>(src2), reinterpret_cast<uchar4*>(dst), step, width, size.height);
	}
}

void launchAddCKernel(const Npp8u* src1, const Npp32u value, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method)
{
	if (method == BASIC)
	{
		dim3 block(32, 32);
		dim3 grid((size.width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		addc << <grid, block, 0, stream >> > (src1, static_cast<Npp8u>(value), dst, step, size.width, size.height);
	}
	else if (method == M_4K)
	{
		auto width = size.width >> 2;
		step = step >> 2;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);

		const Npp32u simdvalue = value | value << 8 | value << 16 | value << 24;

		addc_4k << <grid, block, 0, stream >> > (reinterpret_cast<const uint32_t*>(src1), simdvalue, reinterpret_cast<uint32_t*>(dst), step, width, size.height);
	}
}

void launchMulCKernel(const Npp8u* src1, const Npp32u value, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method)
{
	if (method == BASIC)
	{
		dim3 block(32, 32);
		dim3 grid((size.width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		mulc << <grid, block, 0, stream >> > (src1, static_cast<Npp8u>(value), dst, step, size.width, size.height);
	}
	else if (method == M_4K)
	{
		auto width = size.width >> 2;
		step = step >> 2;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);

		mulc_4k << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(src1), value, reinterpret_cast<uchar4*>(dst), step, width, size.height);
	}
}

void launchAddCMulCKernel(const Npp8u* src, const Npp32u addValue, const Npp32f mulValue, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method)
{
	if (method == BASIC)
	{
		dim3 block(32, 32);
		dim3 grid((size.width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		addcmulc << <grid, block, 0, stream >> > (src, static_cast<Npp8u>(addValue), static_cast<Npp8u>(mulValue), dst, step, size.width, size.height);
	}
	else if (method == M_4K)
	{
		auto width = size.width >> 2;
		step = step >> 2;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);

		addcmulc_4k << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(src), addValue, mulValue, reinterpret_cast<uchar4*>(dst), step, width, size.height);
	}
}

void launchBrightnessContrast_uv_int8(const Npp8u* src, const Npp32s addValue, const Npp32f mulValue, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method)
{
	if (method == BASIC)
	{
	
	}
	else if (method == M_4K)
	{
		auto width = size.width >> 2;
		step = step >> 2;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);

		brightnesscontrast_uv_int8_4k << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(src), addValue, mulValue, reinterpret_cast<uchar4*>(dst), step, width, size.height);
	}
}