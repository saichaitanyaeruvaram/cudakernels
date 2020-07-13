#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernels.h"

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
}

__global__ void add_32(const Npp8u* src1, const Npp8u* src2, Npp8u* dst, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	for (auto i = 0; i < 64; i++)
	{
		int offset = y * step + x + i;
		dst[offset] = src1[offset] + src2[offset];
	}
}

void launchAddKernel(const Npp8u* src1, const Npp8u* src2, Npp8u* dst, int step, NppiSize size, cudaStream_t stream, std::string method)
{
	if (method == ADD_BASIC)
	{
		dim3 block(64, 64);
		dim3 grid((size.width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		add << <grid, block, 0, stream >> > (src1, src2, dst, step, size.width, size.height);
	}
	else if (method == ADD_32)
	{ 
		auto width = size.width >> 6;
		dim3 block(64, 64);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		add_32 << <grid, block, 0, stream >> > (src1, src2, dst, step, width, size.height);
	}
}