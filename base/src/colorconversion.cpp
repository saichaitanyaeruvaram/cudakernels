#include "tests.h"
#include "kernels.h"
#include <vector>


std::vector<std::vector<uint8_t>> yuvvalues;
std::vector<std::vector<uint8_t>> rgbvalues_npp;
std::vector<std::vector<uint8_t>> yuvvalues_npp;
std::vector<std::vector<uint8_t>> rgbvalues;
std::vector<std::vector<uint8_t>> yuvvalues_custom;

#define CLAMP_255(x) x < 0 ? 0 : (x > 255 ? 255 : x)
#define CLAMP_int8(x) x < -128 ? -128 : (x > 127 ? 127 : x)

#define YUV_TO_RGB( Y, U, V, R, G, B )                      \
    do                                                      \
    {                                                       \
		int rTmp = Y + (1.370705 * (V-128));                      \
        int gTmp = Y - (0.698001 * (V-128)) - (0.337633 * (U-128));     \
        int bTmp = Y + (1.732446 * (U-128));                      \
        R = CLAMP_255(rTmp);                                \
        G = CLAMP_255(gTmp);                                \
        B = CLAMP_255(bTmp);                                \
	} while (0)

#define RGB_TO_YUV( R, G, B, Y, U, V )                      \
    do                                                      \
    {                                                       \
		int yTmp = 0.299*R + 0.587*G + 0.114*B;				\
		int uTmp = 0.436*B - 0.289*G - 0.147*R;				\
		int vTmp = 0.615*R - 0.515*G - 0.1*B;				\
		Y = CLAMP_255(yTmp);	\
		uTmp = CLAMP_int8(uTmp); \
		U = uTmp + 128;     					\
        V = (128) + (CLAMP_int8(vTmp));							\
	} while (0)

void init_test_values()
{
	yuvvalues.push_back({ 255, 128, 128 });
	yuvvalues_npp.push_back({ 255, 128, 128 });
	rgbvalues_npp.push_back({ 255, 255, 255 });

	yuvvalues.push_back({ 0, 128, 128 });
	yuvvalues_npp.push_back({ 0, 128, 128 });
	rgbvalues_npp.push_back({ 0, 0, 0 });

	yuvvalues.push_back({ 76, 84, 255 });
	yuvvalues_npp.push_back({ 76, 90, 253 });
	rgbvalues_npp.push_back({ 220, 19, 0 });

	yuvvalues.push_back({ 225, 0, 148 });
	yuvvalues_npp.push_back({ 223, 18, 148 });
	rgbvalues_npp.push_back({ 247, 255, 0 });

	yuvvalues.push_back({ 105, 212, 234 });
	yuvvalues_npp.push_back({ 102, 203, 235 });
	rgbvalues_npp.push_back({ 225, 10, 255 });

	yuvvalues.push_back({ 29, 255, 107 });
	yuvvalues_npp.push_back({ 30, 238, 105 });
	rgbvalues_npp.push_back({ 5, 0, 255 });

	for (auto i = 0; i < yuvvalues.size(); i++)
	{
		rgbvalues.push_back({ 0, 0, 0 });
		yuvvalues_custom.push_back({ 0, 0, 0 });
		YUV_TO_RGB(yuvvalues[i][0],
			yuvvalues[i][1],
			yuvvalues[i][2],
			rgbvalues[i][0],
			rgbvalues[i][1],
			rgbvalues[i][2]
		);

		RGB_TO_YUV(
			rgbvalues_npp[i][0],
			rgbvalues_npp[i][1],
			rgbvalues_npp[i][2],
			yuvvalues_custom[i][0],
			yuvvalues_custom[i][1],
			yuvvalues_custom[i][2]
		);

		std::cout << "rgb_diff<>" << abs(rgbvalues[i][0] - rgbvalues_npp[i][0]) << "<>" << abs(rgbvalues[i][1] - rgbvalues_npp[i][1]) << "<>" << abs(rgbvalues[i][2] - rgbvalues_npp[i][2]) << std::endl;
		std::cout << "yuv_diff<>" << abs(yuvvalues_custom[i][0] - yuvvalues_npp[i][0]) << "<>" << abs(yuvvalues_custom[i][1] - yuvvalues_npp[i][1]) << "<>" << abs(yuvvalues_custom[i][2] - yuvvalues_npp[i][2]) << std::endl;
	}

}

void testYUV420ToRGB(int argc, char **argv)
{
	std::string method = "";
	if (argc == 1)
	{
		method = argv[0];
	}

	init_test_values();

	DeviceBuffer y, u, v, r, g, b;

	int width = 1920;
	int height = 1080;

	int width_2 = width >> 1;
	int height_2 = height >> 1;

	getDeviceBuffer(width, height, 0, y);
	getDeviceBuffer(width_2, height_2, 0, u);
	getDeviceBuffer(width_2, height_2, 0, v);
	getDeviceBuffer(width, height, 0, r);
	getDeviceBuffer(width, height, 0, g);
	getDeviceBuffer(width, height, 0, b);

	NppiSize size = { width, height };
	int step_y = y.step();
	int step_uv = u.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	auto y8u = static_cast<uint8_t *>(y.data());
	auto u8u = static_cast<uint8_t *>(u.data());
	auto v8u = static_cast<uint8_t *>(v.data());
	auto r8u = static_cast<uint8_t *>(r.data());
	auto g8u = static_cast<uint8_t *>(g.data());
	auto b8u = static_cast<uint8_t *>(b.data());

	for (auto i = 0; i < yuvvalues.size(); i++)
	{
		auto yuv = yuvvalues[i];
		y.memset(yuv[0]);
		u.memset(yuv[1]);
		v.memset(yuv[2]);
		launch_yuv420torgb(y8u, u8u, v8u, r8u, g8u, b8u, step_y, step_uv, size, stream, method);
		ck(cudaStreamSynchronize(stream));
		std::cout << static_cast<int>(yuv[0]) << "<>" << static_cast<int>(yuv[1]) << "<>" << static_cast<int>(yuv[2]) << std::endl;
		std::cout << "checking r<>";
		copyAndCheckValue(r, rgbvalues[i][0]);
		std::cout << "checking g<>";
		copyAndCheckValue(g, rgbvalues[i][1]);
		std::cout << "checking b<>" << std::endl;
		copyAndCheckValue(b, rgbvalues[i][2]);
	}

	profile([&]() {
		launch_yuv420torgb(y8u, u8u, v8u, r8u, g8u, b8u, step_y, step_uv, size, stream, method);

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));

}

void testRGBToYUV420(int argc, char **argv)
{
	std::string method = "";
	if (argc == 1)
	{
		method = argv[0];
	}

	init_test_values();

	DeviceBuffer y, u, v, r, g, b;

	int width = 1920;
	int height = 1080;
	int width_2 = width >> 1;
	int height_2 = height >> 1;

	getDeviceBuffer(width, height, 0, y);
	getDeviceBuffer(width_2, height_2, 0, u);
	getDeviceBuffer(width_2, height_2, 0, v);
	getDeviceBuffer(width, height, 0, r);
	getDeviceBuffer(width, height, 0, g);
	getDeviceBuffer(width, height, 0, b);

	NppiSize size = { width, height };
	int step_y = y.step();
	int step_uv = u.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto y8u = static_cast<uint8_t *>(y.data());
	auto u8u = static_cast<uint8_t *>(u.data());
	auto v8u = static_cast<uint8_t *>(v.data());
	auto r8u = static_cast<uint8_t *>(r.data());
	auto g8u = static_cast<uint8_t *>(g.data());
	auto b8u = static_cast<uint8_t *>(b.data());

	for (auto i = 0; i < yuvvalues.size(); i++)
	{
		r.memset(rgbvalues_npp[i][0]);
		g.memset(rgbvalues_npp[i][1]);
		b.memset(rgbvalues_npp[i][2]);
		launch_rgbtoyuv420(r8u, g8u, b8u, y8u, u8u, v8u, step_y, step_uv, size, stream, method);
		ck(cudaStreamSynchronize(stream));
		std::cout << static_cast<int>(rgbvalues_npp[i][0]) << "<>" << static_cast<int>(rgbvalues_npp[i][1]) << "<>" << static_cast<int>(rgbvalues_npp[i][2]) << std::endl;
		std::cout << "checking y<>";
		copyAndCheckValue(y, yuvvalues_custom[i][0]);
		std::cout << "checking u<>";
		copyAndCheckValue(u, yuvvalues_custom[i][1]);
		std::cout << "checking v<>" << std::endl;
		copyAndCheckValue(v, yuvvalues_custom[i][2]);
	}

	profile([&]() {
		launch_rgbtoyuv420(r8u, g8u, b8u, y8u, u8u, v8u, step_y, step_uv, size, stream, method);

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));

}

void testYUV420ToRGBNPP()
{
	init_test_values();

	DeviceBuffer y, u, v, r, g, b;

	int width = 1920;
	int height = 1080;
	int width_2 = width >> 1;
	int height_2 = height >> 1;

	getDeviceBuffer(width, height, 0, y);
	getDeviceBuffer(width_2, height_2, 0, u);
	getDeviceBuffer(width_2, height_2, 0, v);
	getDeviceBuffer(width, height, 0, r);
	getDeviceBuffer(width, height, 0, g);
	getDeviceBuffer(width, height, 0, b);

	NppiSize size = { width, height };
	int step_y = y.step();
	int step_uv = u.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto y8u = static_cast<uint8_t *>(y.data());
	auto u8u = static_cast<uint8_t *>(u.data());
	auto v8u = static_cast<uint8_t *>(v.data());
	auto r8u = static_cast<uint8_t *>(r.data());
	auto g8u = static_cast<uint8_t *>(g.data());
	auto b8u = static_cast<uint8_t *>(b.data());

	const Npp8u* const pSrc[3] = { y8u, u8u, v8u };
	int rSrcStep[3] = { step_y, step_uv, step_uv };
	Npp8u* pDst[3] = { r8u, g8u, b8u };

	for (auto i = 0; i < yuvvalues.size(); i++)
	{
		auto yuv = yuvvalues[i];
		y.memset(yuv[0]);
		u.memset(yuv[1]);
		v.memset(yuv[2]);
		check_nppstatus(nppiYUV420ToRGB_8u_P3R_Ctx(pSrc,
			rSrcStep,
			pDst,
			step_y,
			size,
			nppStreamCtx
		));
		ck(cudaStreamSynchronize(stream));
		std::cout << static_cast<int>(yuv[0]) << "<>" << static_cast<int>(yuv[1]) << "<>" << static_cast<int>(yuv[2]) << std::endl;
		std::cout << "checking r<>";
		copyAndCheckValue(r, rgbvalues_npp[i][0]);
		std::cout << "checking g<>";
		copyAndCheckValue(g, rgbvalues_npp[i][1]);
		std::cout << "checking b<>" << std::endl;
		copyAndCheckValue(b, rgbvalues_npp[i][2]);
	}

	profile([&]() {
		check_nppstatus(nppiYUV420ToRGB_8u_P3R_Ctx(pSrc,
			rSrcStep,
			pDst,
			step_y,
			size,
			nppStreamCtx
		));

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));
}

void testRGBToYUV420NPP()
{
	init_test_values();

	DeviceBuffer y, u, v, r, g, b;

	int width = 1920;
	int height = 1080;
	int width_2 = width >> 1;
	int height_2 = height >> 1;

	getDeviceBuffer(width, height, 0, y);
	getDeviceBuffer(width_2, height_2, 0, u);
	getDeviceBuffer(width_2, height_2, 0, v);
	getDeviceBuffer(width, height, 0, r);
	getDeviceBuffer(width, height, 0, g);
	getDeviceBuffer(width, height, 0, b);

	NppiSize size = { width, height };
	int step_y = y.step();
	int step_uv = u.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto y8u = static_cast<uint8_t *>(y.data());
	auto u8u = static_cast<uint8_t *>(u.data());
	auto v8u = static_cast<uint8_t *>(v.data());
	auto r8u = static_cast<uint8_t *>(r.data());
	auto g8u = static_cast<uint8_t *>(g.data());
	auto b8u = static_cast<uint8_t *>(b.data());

	const Npp8u* const pSrc[3] = { r8u, g8u, b8u };
	int rDstStep[3] = { step_y, step_uv, step_uv };
	Npp8u* pDst[3] = { y8u, u8u, v8u };

	for (auto i = 0; i < yuvvalues.size(); i++)
	{
		r.memset(rgbvalues_npp[i][0]);
		g.memset(rgbvalues_npp[i][1]);
		b.memset(rgbvalues_npp[i][2]);
		check_nppstatus(nppiRGBToYUV420_8u_P3R_Ctx(pSrc,
			step_y,
			pDst,
			rDstStep,
			size,
			nppStreamCtx
		));
		ck(cudaStreamSynchronize(stream));
		std::cout << static_cast<int>(rgbvalues_npp[i][0]) << "<>" << static_cast<int>(rgbvalues_npp[i][1]) << "<>" << static_cast<int>(rgbvalues_npp[i][2]) << std::endl;
		std::cout << "checking y<>";
		copyAndCheckValue(y, yuvvalues_npp[i][0]);
		std::cout << "checking u<>";
		copyAndCheckValue(u, yuvvalues_npp[i][1]);
		std::cout << "checking v<>" << std::endl;
		copyAndCheckValue(v, yuvvalues_npp[i][2]);
	}

	profile([&]() {
		check_nppstatus(nppiRGBToYUV420_8u_P3R_Ctx(pSrc,
			step_y,
			pDst,
			rDstStep,
			size,
			nppStreamCtx
		));

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));
}

void testYUV420HueSaturation(int argc, char **argv)
{
	std::string method = "";
	if (argc == 1)
	{
		method = argv[0];
	}

	init_test_values();

	DeviceBuffer y, u, v, Y, U, V;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 0, y);
	getDeviceBuffer(width, height, 0, u);
	getDeviceBuffer(width, height, 0, v);
	getDeviceBuffer(width, height, 0, Y);
	getDeviceBuffer(width, height, 0, U);
	getDeviceBuffer(width, height, 0, V);

	auto step_y = y.step();
	auto step_uv = u.step();
	NppiSize size = { width, height };

	auto y8u = static_cast<uint8_t *>(y.data());
	auto u8u = static_cast<uint8_t *>(u.data());
	auto v8u = static_cast<uint8_t *>(v.data());
	auto Y8u = static_cast<uint8_t *>(Y.data());
	auto U8u = static_cast<uint8_t *>(U.data());
	auto V8u = static_cast<uint8_t *>(V.data());

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	profile([&]() {
		launch_yuv420huesaturation(y8u, u8u, v8u, Y8u, U8u, V8u, 0.1, 0.1, step_y, step_uv, size, stream, method);

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));
}

void testRGBHueSaturation(int argc, char **argv)
{
	std::string method = "";
	if (argc == 1)
	{
		method = argv[0];
	}

	init_test_values();

	DeviceBuffer r, g, b, R, G, B;

	int width = 1920;
	int height = 1080;

	getDeviceBuffer(width, height, 0, r);
	getDeviceBuffer(width, height, 0, g);
	getDeviceBuffer(width, height, 0, b);
	getDeviceBuffer(width, height, 0, R);
	getDeviceBuffer(width, height, 0, G);
	getDeviceBuffer(width, height, 0, B);

	auto step = r.step();
	NppiSize size = { width, height };

	auto r8u = static_cast<uint8_t *>(r.data());
	auto g8u = static_cast<uint8_t *>(g.data());
	auto b8u = static_cast<uint8_t *>(b.data());
	auto R8u = static_cast<uint8_t *>(R.data());
	auto G8u = static_cast<uint8_t *>(G.data());
	auto B8u = static_cast<uint8_t *>(B.data());

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	profile([&]() {
		launch_rgbhuesaturation(r8u, g8u, b8u, R8u, G8u, B8u, 0.1, 0.1, step, size, stream, method);

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));
}

void testRGBToHSVNPP()
{
	init_test_values();

	DeviceBuffer hsv, rgb;

	int width = 1920;
	int row_width = width*3;
	int height = 1080;

	getDeviceBuffer(row_width, height, 0, hsv);
	getDeviceBuffer(row_width, height, 0, rgb);

	NppiSize size = { width, height };
	int step = hsv.step();

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));

	NppStreamContext nppStreamCtx;
	nppStreamCtx.hStream = stream;

	auto hsv8u = static_cast<uint8_t *>(hsv.data());
	auto rgb8u = static_cast<uint8_t *>(rgb.data());

	profile([&]() {
		check_nppstatus(nppiRGBToHSV_8u_C3R_Ctx(hsv8u,
			step,
			rgb8u,
			step,
			size,
			nppStreamCtx
		));

		ck(cudaStreamSynchronize(stream));
	});

	ck(cudaStreamDestroy(stream));
}