#include "common.h"
#include "kernels.h"
#include "npp.h"

void testAdd()
{
	void* src1;
	void* src2;
	void* dst;

	int width = 1920;
	int height = 1080;
	int step = 0;

	getDeviceBuffer(width, height, 100, src1, step);
	getDeviceBuffer(width, height, 50, src2, step);
	getDeviceBuffer(width, height, 0, dst, step);

	NppiSize size = { width, height };

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	auto src18u = static_cast<uint8_t*>(src1);
	auto src28u = static_cast<uint8_t*>(src2);
	auto dst8u = static_cast<uint8_t*>(dst);

	profile([&]() {
		launchAddKernel(src18u, src28u, dst8u, step, size, stream);
		cudaStreamSynchronize(stream);
	});

	ck(cudaStreamDestroy(stream));
}

void testAddNPP()
{
	void* src1;
	void* src2;
	void* dst;

	int width = 1920;
	int height = 1080;
	int step = 0;

	getDeviceBuffer(width, height, 100, src1, step);
	getDeviceBuffer(width, height, 50, src2, step);
	getDeviceBuffer(width, height, 0, dst, step);

	NppiSize size = { width, height };

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext 	nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto src18u = static_cast<uint8_t*>(src1);
	auto src28u = static_cast<uint8_t*>(src2);
	auto dst8u = static_cast<uint8_t*>(dst);

	profile([&]() {
		nppiAdd_8u_C1RSfs_Ctx(src18u,
			step,
			src28u,
			step,
			dst8u,
			step,
			size,
			0,
			nppStreamCtx);

		cudaStreamSynchronize(stream);
	});

	ck(cudaStreamDestroy(stream));
}