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

	auto src18u = static_cast<uint8_t *>(src1.data());
	auto src28u = static_cast<uint8_t *>(src2.data());
	auto dst8u = static_cast<uint8_t *>(dst.data());

	auto method = BASIC;
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

void testAddC(int argc, char **argv)
{
	DeviceBuffer src;
	DeviceBuffer dst;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 250, src);
	getDeviceBuffer(width, height, 0, dst);

	NppiSize size = { width, height };
	int step = src.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto src8u = static_cast<uint8_t *>(src.data());
	auto dst8u = static_cast<uint8_t *>(dst.data());

	auto method = BASIC;
	if (argc == 1)
	{
		method = argv[0];
	}

	launchAddCKernel(src8u, 50, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);

	copyAndCheckValue(dst, 255);

	launchAddCKernel(src8u, 1, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);

	copyAndCheckValue(dst, 251);

	profile([&]() {
		launchAddCKernel(src8u, 50, dst8u, step, size, stream, method);
		cudaStreamSynchronize(stream);
	});

	ck(cudaStreamDestroy(stream));
}

void testAddCMulC(int argc, char **argv)
{
	DeviceBuffer src;
	DeviceBuffer dst;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 50, src);
	getDeviceBuffer(width, height, 0, dst);

	NppiSize size = { width, height };
	int step = src.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto src8u = static_cast<uint8_t *>(src.data());
	auto dst8u = static_cast<uint8_t *>(dst.data());

	auto method = BASIC;
	if (argc == 1)
	{
		method = argv[0];
	}

	launchAddCMulCKernel(src8u, 50, 2, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);

	copyAndCheckValue(dst, 150);

	launchAddCMulCKernel(src8u, 250, 1, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);

	copyAndCheckValue(dst, 255);

	launchAddCMulCKernel(src8u, 0, 6, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);

	copyAndCheckValue(dst, 255);

	profile([&]() {
		launchAddCMulCKernel(src8u, 50, 2, dst8u, step, size, stream, method);
		cudaStreamSynchronize(stream);
	});

	ck(cudaStreamDestroy(stream));
}

void testBrightnessContrast_uv_int8(int argc, char **argv)
{
	DeviceBuffer src;
	DeviceBuffer dst;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 134, src);
	getDeviceBuffer(width, height, 0, dst);

	NppiSize size = { width, height };
	int step = src.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto src8u = static_cast<uint8_t *>(src.data());
	auto dst8u = static_cast<uint8_t *>(dst.data());

	auto method = M_4K;
	if (argc == 1)
	{
		method = argv[0];
	}

	src.memset(134);
	launchBrightnessContrast_uv_int8(src8u, 50, 0.5, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);
	copyAndCheckValue(dst, 131);
	src.memset(122);
	launchBrightnessContrast_uv_int8(src8u, 50, 0.5, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);
	copyAndCheckValue(dst, 125);

	// > 127
	src.memset(128+90);
	launchBrightnessContrast_uv_int8(src8u, 50, 3, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);
	copyAndCheckValue(dst, 255);
	
	// < -128
	src.memset(128 - 90);
	launchBrightnessContrast_uv_int8(src8u, 50, 3, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);
	copyAndCheckValue(dst, 0);

	// > -128
	src.memset(128 - 30);
	launchBrightnessContrast_uv_int8(src8u, -100, 3, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);
	copyAndCheckValue(dst, 38);

	profile([&]() {
		launchBrightnessContrast_uv_int8(src8u, 50, 2, dst8u, step, size, stream, method);
		cudaStreamSynchronize(stream);
	});

	ck(cudaStreamDestroy(stream));
}

void testMulC(int argc, char **argv)
{
	DeviceBuffer src;
	DeviceBuffer dst;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 100, src);
	getDeviceBuffer(width, height, 0, dst);

	NppiSize size = { width, height };
	int step = src.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto src8u = static_cast<uint8_t *>(src.data());
	auto dst8u = static_cast<uint8_t *>(dst.data());

	auto method = BASIC;
	if (argc == 1)
	{
		method = argv[0];
	}

	launchMulCKernel(src8u, 2, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);

	copyAndCheckValue(dst, 200);

	launchMulCKernel(src8u, 3, dst8u, step, size, stream, method);
	cudaStreamSynchronize(stream);

	copyAndCheckValue(dst, 255);

	profile([&]() {
		launchMulCKernel(src8u, 2, dst8u, step, size, stream, method);
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

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto src18u = static_cast<uint8_t *>(src1.data());
	auto src28u = static_cast<uint8_t *>(src2.data());
	auto dst8u = static_cast<uint8_t *>(dst.data());

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

void testAddCNPP()
{
	DeviceBuffer src;
	DeviceBuffer dst;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 250, src);
	getDeviceBuffer(width, height, 0, dst);

	NppiSize size = { width, height };
	int step = src.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto src8u = static_cast<uint8_t *>(src.data());
	auto dst8u = static_cast<uint8_t *>(dst.data());

	check_nppstatus(nppiAddC_8u_C1IRSfs_Ctx(50,
		src8u,
		step,
		size,
		0,
		nppStreamCtx));

	ck(cudaStreamSynchronize(stream));
	copyAndCheckValue(src, 255);

	profile([&]() {
		check_nppstatus(nppiAddC_8u_C1RSfs_Ctx(src8u,
			step,
			50,
			dst8u,
			step,
			size,
			0,
			nppStreamCtx));

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));
}

void testAddCMulCNPP()
{
	DeviceBuffer src;
	DeviceBuffer dst1;
	DeviceBuffer dst2;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 150, src);
	getDeviceBuffer(width, height, 0, dst1);
	getDeviceBuffer(width, height, 0, dst2);

	NppiSize size = { width, height };
	int step = src.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto src8u = static_cast<uint8_t *>(src.data());
	auto dst18u = static_cast<uint8_t *>(dst1.data());
	auto dst28u = static_cast<uint8_t *>(dst2.data());

	check_nppstatus(nppiMulC_8u_C1IRSfs_Ctx(2,
		src8u,
		step,
		size,
		0,
		nppStreamCtx));

	check_nppstatus(nppiAddC_8u_C1IRSfs_Ctx(50,
		src8u,
		step,
		size,
		0,
		nppStreamCtx));

	ck(cudaStreamSynchronize(stream));
	copyAndCheckValue(src, 255);

	profile([&]() {
		check_nppstatus(nppiMulC_8u_C1IRSfs_Ctx(2,
			src8u,
			step,
			size,
			0,
			nppStreamCtx));

		check_nppstatus(nppiAddC_8u_C1IRSfs_Ctx(50,
			src8u,
			step,
			size,
			0,
			nppStreamCtx));

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));
}

void testMulCNPP()
{
	DeviceBuffer src;
	DeviceBuffer dst;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 100, src);
	getDeviceBuffer(width, height, 0, dst);

	NppiSize size = { width, height };
	int step = src.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto src8u = static_cast<uint8_t *>(src.data());
	auto dst8u = static_cast<uint8_t *>(dst.data());

	check_nppstatus(nppiMulC_8u_C1IRSfs_Ctx(2,
		src8u,
		step,
		size,
		0,
		nppStreamCtx));

	ck(cudaStreamSynchronize(stream));
	copyAndCheckValue(src, 200);

	profile([&]() {
		check_nppstatus(nppiMulC_8u_C1IRSfs_Ctx(2,
			src8u,
			step,
			size,
			0,
			nppStreamCtx));

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));
}

void testOverlayYUV()
{
	DeviceBuffer Y, U, V;
	DeviceBuffer overlay_y, overlay_u, overlay_v;
	DeviceBuffer Yout, Uout, Vout;

	int width = 1920;
	int height = 1080;

	int width_uv = width >> 1;
	int height_uv = height >> 1;

	getDeviceBuffer(width, height, 30, Y);
	getDeviceBuffer(width_uv, height_uv, 148, U);
	getDeviceBuffer(width_uv, height_uv, 118, V);
	getDeviceBuffer(width, height, 200, overlay_y);
	getDeviceBuffer(width_uv, height_uv, 178, overlay_u);
	getDeviceBuffer(width_uv, height_uv, 108, overlay_v);
	getDeviceBuffer(width, height, 200, Yout);
	getDeviceBuffer(width_uv, height_uv, 200, Uout);
	getDeviceBuffer(width_uv, height_uv, 200, Vout);

	NppiSize size = { width, height };
	int step_y = Y.step();
	int step_uv = U.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	const uchar4* src[3] = {static_cast<uchar4 *>(Y.data()), static_cast<uchar4 *>(U.data()), static_cast<uchar4 *>(V.data()),};
	const uchar4* overlay[3] = {static_cast<uchar4 *>(overlay_y.data()), static_cast<uchar4 *>(overlay_u.data()), static_cast<uchar4 *>(overlay_v.data()),};
	uchar4* dst[3] = {static_cast<uchar4 *>(Yout.data()), static_cast<uchar4 *>(Uout.data()), static_cast<uchar4 *>(Vout.data()),};

	launchYUVOverlayKernel(src, overlay, dst, 0.5, step_y, step_uv, size, stream);
	cudaStreamSynchronize(stream);

	copyAndCheckValue(Yout, 130);	
	copyAndCheckValue(Uout, 173);
	copyAndCheckValue(Vout, 108);

	profile([&]() {
		launchYUVOverlayKernel(src, overlay, dst, 0.5, step_y, step_uv, size, stream);
		cudaStreamSynchronize(stream);
	});

	ck(cudaStreamDestroy(stream));	
}