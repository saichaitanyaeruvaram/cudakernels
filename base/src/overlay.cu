#include "kernels.h"

#define CLAMP_255(x) x < 0 ? 0 : (x > 255 ? 255 : x)
#define CLAMP_INT8(x) x < -128 ? -128 : (x > 127 ? 127 : x)

#define UV_OVERLAY(src_pixel, overlay_pixel, dst_pixel, alpha)                      \
do                                                                                  \
{                                                                                   \
    Npp32f temp = __fmul_rn(__fsub_rn(overlay_pixel, 128), alpha);                  \
    temp = __fadd_rn(__fsub_rn(src_pixel, 128), temp);                              \
    dst_pixel = 128 + (CLAMP_INT8(temp));                                           \
} while(0)

__global__ void uvOverlayKernel(const uchar4* src, const uchar4* overlay, uchar4* dst, float alpha, int step, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
    UV_OVERLAY(src[offset].x, overlay[offset].x, dst[offset].x, alpha);
    UV_OVERLAY(src[offset].y, overlay[offset].y, dst[offset].y, alpha);
    UV_OVERLAY(src[offset].z, overlay[offset].z, dst[offset].z, alpha);
    UV_OVERLAY(src[offset].w, overlay[offset].w, dst[offset].w, alpha);
}

void launchUVOverlayKernel(const Npp8u* src, const Npp8u* overlay, Npp8u* dst, Npp32f alpha, int step, NppiSize size, cudaStream_t stream)
{
    auto width = size.width >> 2;
	step = step >> 2;
	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
	uvOverlayKernel << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(src), reinterpret_cast<const uchar4*>(overlay), reinterpret_cast<uchar4*>(dst), alpha, step, width, size.height);
}