#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernels.h"
#include <iostream>

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

__global__ void addc(const Npp8u* src1, const Npp8u value, Npp8u* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	dst[offset] = src1[offset] + value;
	dst[offset] |= -(dst[offset] < value);
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

__global__ void add_4k(const Npp32u* src1, const Npp32u* src2, Npp32u* dst, int step, int width, int height)
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

__global__ void addc_4k(const Npp32u* src1, const Npp32u value, Npp32u* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	dst[offset] = __vaddus4(src1[offset], value);	
}

void launchAddKernel(const Npp8u* src1, const Npp8u* src2, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method)
{
	if (method == ADD_BASIC)
	{
		dim3 block(32, 32);
		dim3 grid((size.width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		add << <grid, block, 0, stream >> > (src1, src2, dst, step, size.width, size.height);
	}
	else if (method == ADD_32)
	{ 
		auto width = size.width >> 5;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		add_32 << <grid, block, 0, stream >> > (src1, src2, dst, step, width, size.height);
	}
	else if (method == ADD_4K)
	{
		auto width = size.width >> 2;
		step = step >> 2;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		
		add_4k << <grid, block, 0, stream >> > (reinterpret_cast<const uint32_t*>(src1), reinterpret_cast<const uint32_t*>(src2), reinterpret_cast<uint32_t*>(dst), step, width, size.height);
	}
}

void launchAddCKernel(const Npp8u* src1, const Npp32u value, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method)
{
	if (method == ADD_BASIC)
	{
		dim3 block(32, 32);
		dim3 grid((size.width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		addc << <grid, block, 0, stream >> > (src1, static_cast<Npp8u>(value), dst, step, size.width, size.height);
	}
	else if (method == ADD_4K)
	{
		auto width = size.width >> 2;
		step = step >> 2;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);

		const Npp32u simdvalue = value | value << 8 | value << 16 | value << 24;
		
		addc_4k << <grid, block, 0, stream >> > (reinterpret_cast<const uint32_t*>(src1), simdvalue, reinterpret_cast<uint32_t*>(dst), step, width, size.height);
	}
}