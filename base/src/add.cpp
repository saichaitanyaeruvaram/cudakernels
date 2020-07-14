#include "tests.h"
#include "kernels.h"
#include "npp.h"
#include <assert.h>

void testAdd(int argc, char **argv)
{
	DeviceBuffer src1;
	DeviceBuffer src2;
	DeviceBuffer dst;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 100, src1);
	getDeviceBuffer(width, height, 50, src2);
	getDeviceBuffer(width, height, 0, dst);

	NppiSize size = { width, height };
	int step = src1.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	auto src18u = static_cast<uint8_t*>(src1.data());
	auto src28u = static_cast<uint8_t*>(src2.data());
	auto dst8u = static_cast<uint8_t*>(dst.data());

	auto method = ADD_BASIC;
	if (argc == 1)
	{
		method = argv[0];
	}

	launchAddKernel(src18u, src28u, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);

	copyAndCheckValue(dst, 150);

	profile([&]() {
		launchAddKernel(src18u, src28u, dst8u, step, size, stream, method);
		cudaStreamSynchronize(stream);
	});

	ck(cudaStreamDestroy(stream));
}

void testAddNPP()
{
	DeviceBuffer src1;
	DeviceBuffer src2;
	DeviceBuffer dst;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 100, src1);
	getDeviceBuffer(width, height, 50, src2);
	getDeviceBuffer(width, height, 0, dst);

	NppiSize size = { width, height };
	int step = src1.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext 	nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto src18u = static_cast<uint8_t*>(src1.data());
	auto src28u = static_cast<uint8_t*>(src2.data());
	auto dst8u = static_cast<uint8_t*>(dst.data());

	check_nppstatus(nppiAdd_8u_C1RSfs_Ctx(src18u,
		step,
		src28u,
		step,
		dst8u,
		step,
		size,
		0,
		nppStreamCtx));

	ck(cudaStreamSynchronize(stream));
	copyAndCheckValue(dst, 150);

	profile([&]() {
		check_nppstatus(nppiAdd_8u_C1RSfs_Ctx(src18u,
			step,
			src28u,
			step,
			dst8u,
			step,
			size,
			0,
			nppStreamCtx));

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));
}