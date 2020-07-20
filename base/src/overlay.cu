#include "kernels.h"

#define CLAMP_255(x) x < 0 ? 0 : (x > 255 ? 255 : x)
#define CLAMP_INT8(x) x < -128 ? -128 : (x > 127 ? 127 : x)

#define OVERLAY(src_pixel, overlay_pixel, dst_pixel, alpha)                         \
do                                                                                  \
{                                                                                   \
    Npp32f temp = __fmul_rn(overlay_pixel, alpha);                                  \
    temp = __fadd_rn(src_pixel, temp);                                              \
    dst_pixel = CLAMP_255(temp);                                                    \
} while(0)

#define UV_OVERLAY(src_pixel, overlay_pixel, dst_pixel, alpha)                      \
do                                                                                  \
{                                                                                   \
    Npp32f temp = __fmul_rn(__fsub_rn(overlay_pixel, 128), alpha);                  \
    temp = __fadd_rn(__fsub_rn(src_pixel, 128), temp);                              \
    dst_pixel = 128 + (CLAMP_INT8(temp));                                           \
} while(0)

__global__ void yuvOverlayKernel(const uchar4* Y, const uchar4* U, const uchar4* V, const uchar4* overlay_y, const uchar4* overlay_u, const uchar4* overlay_v, uchar4* Yout, uchar4* Uout, uchar4* Vout, float alpha, int step_y, int step_uv, int width_y, int height_y, int width_uv, int height_uv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width_y || y >= height_y)
	{
		return;
	}

	int offset = y * step_y + x;
	OVERLAY(Y[offset].x, overlay_y[offset].x, Yout[offset].x, alpha);
    OVERLAY(Y[offset].y, overlay_y[offset].y, Yout[offset].y, alpha);
    OVERLAY(Y[offset].z, overlay_y[offset].z, Yout[offset].z, alpha);
    OVERLAY(Y[offset].w, overlay_y[offset].w, Yout[offset].w, alpha);

	
    if(x >= width_uv || y >= height_uv)
    {
        return;
    }
	offset = y * step_uv + x;

    UV_OVERLAY(U[offset].x, overlay_u[offset].x, Uout[offset].x, alpha);
    UV_OVERLAY(U[offset].y, overlay_u[offset].y, Uout[offset].y, alpha);
    UV_OVERLAY(U[offset].z, overlay_u[offset].z, Uout[offset].z, alpha);
    UV_OVERLAY(U[offset].w, overlay_u[offset].w, Uout[offset].w, alpha);

    UV_OVERLAY(V[offset].x, overlay_v[offset].x, Vout[offset].x, alpha);
    UV_OVERLAY(V[offset].y, overlay_v[offset].y, Vout[offset].y, alpha);
    UV_OVERLAY(V[offset].z, overlay_v[offset].z, Vout[offset].z, alpha);
    UV_OVERLAY(V[offset].w, overlay_v[offset].w, Vout[offset].w, alpha);
}

void launchYUVOverlayKernel(const uchar4* src[3], const uchar4* overlay[3], uchar4* dst[3], Npp32f alpha, int step_y, int step_uv, NppiSize size, cudaStream_t stream)
{
    auto width = size.width >> 2;
	step_y = step_y >> 2;
	step_uv = step_uv >> 2;
	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
	yuvOverlayKernel << <grid, block, 0, stream >> > (src[0], src[1], src[2], overlay[0], overlay[1], overlay[2], dst[0], dst[1], dst[2], alpha, step_y, step_uv, width, size.height, width >> 1, size.height >> 1);
}