#include "tests.h"
#include "kernels.h"
#include <vector>
#include <algorithm>


std::vector<std::vector<uint8_t>> yuvvalues;
std::vector<std::vector<uint8_t>> rgbvalues_npp;
std::vector<std::vector<uint8_t>> yuvvalues_npp;
std::vector<std::vector<uint8_t>> rgbvalues;
std::vector<std::vector<uint8_t>> yuvvalues_custom;

#define CLAMP_1(x) x < 0 ? 0 : (x > 1 ? 1 : x)
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
		launch_yuv420huesaturation(y8u, u8u, v8u, Y8u, U8u, V8u, 20, 5, 0.1, 0.1, step_y, step_uv, size, stream, method);

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
	int row_width = width * 3;
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

#define RGB_TO_HSV(R, G, B, H, S, V) 									\
do  																	\
{ 																		\
	Npp32f nNormalizedR = R*0.003921569F; /* 255.0F*/ 					\
	Npp32f nNormalizedG = G*0.003921569F; 								\
	Npp32f nNormalizedB = B*0.003921569F; 								\
	Npp32f nS; 															\
	Npp32f nH; 															\
	/* Value*/ 															\
	Npp32f nV = std::max(nNormalizedR, nNormalizedG); 					\
	nV = std::max(nV, nNormalizedB); 									\
	/*Saturation*/ 														\
	Npp32f nTemp = std::min(nNormalizedR, nNormalizedG); 				\
	nTemp = std::min(nTemp, nNormalizedB); 								\
	Npp32f nDivisor = nV - nTemp; 										\
	if (nV == 0.0F) /*achromatics case*/ 								\
	{ 																	\
		nS = 0.0F; 														\
		nH = 0.0F; 														\
	} 																	\
	else /*chromatics case*/ 											\
	{ 																	\
		nS = nDivisor/nV; 												\
	} 																	\
	/* Hue:*/ 															\
	Npp32f nCr = (nV - nNormalizedR)/ nDivisor; 						\
	Npp32f nCg = (nV - nNormalizedG)/ nDivisor; 						\
	Npp32f nCb = (nV - nNormalizedB)/ nDivisor; 						\
	if (nNormalizedR == nV) 											\
		nH = nCb - nCg; 												\
	else if (nNormalizedG == nV) 										\
		nH = 2.0F + nCr - nCb; 											\
	else if (nNormalizedB == nV) 										\
		nH = 4.0F + nCg - nCr; 											\
	nH = nH*0.166667F; /* 6.0F*/        					\
	if (nH < 0.0F) 														\
		nH = nH + 1.0F; 												\
	H = CLAMP_1(nH); 													\
	S = CLAMP_1(nS); 													\
	V = CLAMP_1(nV); 													\
	 																	\
} while(0)

#define HSV_TO_RGB(nNormalizedH, nNormalizedS, nNormalizedV, R, G, B) 						\
do 																							\
{ 																							\
	Npp32f nR; 																				\
	Npp32f nG; 																				\
	Npp32f nB; 																				\
	if (nNormalizedS == 0.0F) 																\
	{ 																						\
		nR = nG = nB = nNormalizedV; 														\
	} 																						\
	else 																					\
	{ 																						\
		if (nNormalizedH == 1.0F) 															\
			nNormalizedH = 0.0F; 															\
		else 																				\
		{																					\
			/* 0.1667F*/																	\
			nNormalizedH = nNormalizedH*6.0F;  												\
		}																					\
	} 																						\
	Npp32f nI = std::floor(nNormalizedH); 													\
	Npp32f nF = nNormalizedH - nI; 															\
	Npp32f nM = nNormalizedV*(1.0F - nNormalizedS); 										\
	Npp32f nN = nNormalizedV*(1.0F - nNormalizedS*nF); 										\
	Npp32f nK = nNormalizedV*(1.0F - (nNormalizedS*(1.0F - nF)));		 					\
	if (nI == 0.0F) 																		\
	{ 																						\
		nR = nNormalizedV; nG = nK; nB = nM; 												\
	} 																						\
	else if (nI == 1.0F) 																	\
	{ 																						\
		nR = nN; nG = nNormalizedV; nB = nM; 												\
	} 																						\
	else if (nI == 2.0F) 																	\
	{ 																						\
		nR = nM; nG = nNormalizedV; nB = nK; 												\
	} 																						\
	else if (nI == 3.0F) 																	\
	{ 																						\
		nR = nM; nG = nN; nB = nNormalizedV; 												\
	} 																						\
	else if (nI == 4.0F) 																	\
	{ 																						\
		nR = nK; nG = nM; nB = nNormalizedV; 												\
	} 																						\
	else if (nI == 5.0F) 																	\
	{ 																						\
		nR = nNormalizedV; nG = nM; nB = nN; 												\
	} 																						\
	R = CLAMP_255(nR*255.0F); 																\
	G = CLAMP_255(nG*255.0F); 																\
	B = CLAMP_255(nB*255.0F); 																\
																							\
} while(0)

#define RGBHUESATURATIONADJUST(r, g, b, R, G, B, hue, saturation)   \
do 																	\
{ 																	\
	Npp32f H, S, V; 												\
	RGB_TO_HSV(r, g, b, H, S, V); 									\
	H = CLAMP_1(H + hue); 											\
	S = CLAMP_1(S * saturation); 									\
	HSV_TO_RGB(H, S, V, R, G, B);									\
} while(0)

#define YUVHUESATURATIONADJUST(y, u, v, Y, U, V, brightness, contrast, hue, saturation) 	\
do 																							\
{ 																							\
	Npp32f r, g, b; 																		\
	YUV_TO_RGB(y, u, v, r, g, b); 															\
	r = r*contrast + brightness; 															\
	r = CLAMP_255(r);																		\
	g = g * contrast + brightness;															\
	g = CLAMP_255(g);																		\
	b = b * contrast + brightness;															\
	b = CLAMP_255(b);																		\
	Npp8u R, G, B; 																			\
	RGBHUESATURATIONADJUST(r, g, b, R, G, B, hue, saturation); 								\
	RGB_TO_YUV(R, G, B, Y, U, V);															\
} while (0)


#define COMPARE_AND_PRINT(expected_value, actual_value, tolerance, result, messagePrefix)															\
do {																																				\
	result = true;																																	\
	int diff = abs(int(expected_value - actual_value));																								\
	if (diff > tolerance)																															\
	{																																				\
		result = false;																																\
		std::cout << messagePrefix << "<expected_value>" << expected_value << "<actual_value>" << actual_value << "<diff>" << diff << std::endl;	\
	}																																				\
} while(0)



void testYUV42ToRGB_randomvalues(int argc, char **argv)
{
	std::string method = "";
	if (argc == 1)
	{
		method = argv[0];
	}

	DeviceBuffer y, u, v, R, G, B;

	int width = 1920;
	int height = 1080;

	int width_2 = width >> 1;
	int height_2 = height >> 1;

	y.init(width, height);
	u.init(width_2, height_2);
	v.init(width_2, height_2);
	R.init(width, height);
	G.init(width, height);
	B.init(width, height);

	auto step_y = y.step();
	auto step_uv = u.step();
	NppiSize size = { width, height };

	auto y8u = static_cast<uint8_t *>(y.data());
	auto u8u = static_cast<uint8_t *>(u.data());
	auto v8u = static_cast<uint8_t *>(v.data());
	auto R8u = static_cast<uint8_t *>(R.data());
	auto G8u = static_cast<uint8_t *>(G.data());
	auto B8u = static_cast<uint8_t *>(B.data());

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));


	HostBuffer h_y, h_u, h_v, h_R, h_G, h_B;
	h_y.init(width, height);
	h_u.init(width_2, height_2);
	h_v.init(width_2, height_2);
	h_R.init(width, height);
	h_G.init(width, height);
	h_B.init(width, height);
	h_y.setAllValues(30);
	h_u.setAllValues(90);
	h_v.setAllValues(180);
	h_y.copyTo(y);
	h_u.copyTo(u);
	h_v.copyTo(v);


	float hue = 0;
	float saturation = 1;

	R.memset(0);
	G.memset(0);
	B.memset(0);


	launch_yuv420torgb(y8u, u8u, v8u, R8u, G8u, B8u, step_y, step_uv, size, stream, "plain");
	ck(cudaStreamSynchronize(stream));

	h_R.copy(R);
	h_G.copy(G);
	h_B.copy(B);

	auto h_y8u = static_cast<uint8_t *>(h_y.data());
	auto h_u8u = static_cast<uint8_t *>(h_u.data());
	auto h_v8u = static_cast<uint8_t *>(h_v.data());
	auto h_R8u = static_cast<uint8_t *>(h_R.data());
	auto h_G8u = static_cast<uint8_t *>(h_G.data());
	auto h_B8u = static_cast<uint8_t *>(h_B.data());

	bool equal = true;
	for (auto j = 0; j < height && equal; j++)
	{
		auto offset = width * j;
		int offset_uv = width_2 * (j >> 1);
		for (auto i = 0; i < width && equal; i++)
		{
			auto curOffset = offset + i;
			auto curOffset_uv = offset_uv + (i >> 1);
			Npp32u expectedValue_R = 0;
			Npp32u expectedValue_G = 0;
			Npp32u expectedValue_B = 0;
			YUV_TO_RGB(h_y8u[curOffset], h_u8u[curOffset_uv], h_v8u[curOffset_uv], expectedValue_R, expectedValue_G, expectedValue_B);

			auto actualValue_R = static_cast<Npp32u>(h_R8u[curOffset]);
			auto actualValue_G = static_cast<Npp32u>(h_G8u[curOffset]);
			auto actualValue_B = static_cast<Npp32u>(h_B8u[curOffset]);
			//std::cout << j << "<>" << i << "<input>" << static_cast<Npp32u>(h_y8u[curOffset]) << "<>" << static_cast<Npp32u>(h_u8u[curOffset_uv]) << "<>" << static_cast<Npp32u>(h_v8u[curOffset_uv]) << "<output>" << actualValue_R << "<>" << actualValue_G << "<>" << actualValue_B << std::endl;
			if (actualValue_R != expectedValue_R)
			{
				std::cout << j << "<>" << i << "<ouput_r------------------------------------------------------------------------------------------->" << expectedValue_R << "<>" << actualValue_R << std::endl;
				equal = false;
			}
			if (actualValue_G != expectedValue_G)
			{
				std::cout << j << "<>" << i << "<output_g------------------------------------------------------------------------------------------->" << expectedValue_G << "<>" << actualValue_G << std::endl;
				equal = false;
			}
			if (actualValue_B != expectedValue_B)
			{
				std::cout << j << "<>" << i << "<output_b------------------------------------------------------------------------------------------->" << expectedValue_B << "<>" << actualValue_B << std::endl;
				equal = false;
			}
		}
		if (!equal)
		{
			throw "failed";
		}
	}


	y.memset(0);
	u.memset(0);
	v.memset(0);
	h_R.setAllValues(60);
	h_G.setAllValues(120);
	h_B.setAllValues(230);
	h_R.copyTo(R);
	h_G.copyTo(G);
	h_B.copyTo(B);

	launch_rgbtoyuv420(R8u, G8u, B8u, y8u, u8u, v8u, step_y, step_uv, size, stream, "plain");
	ck(cudaStreamSynchronize(stream));

	h_y.copy(y);
	h_u.copy(u);
	h_v.copy(v);

	for (auto j = 0; j < height && equal; j++)
	{
		auto offset = width * j;
		int offset_uv = width_2 * (j >> 1);
		for (auto i = 0; i < width && equal; i++)
		{
			auto curOffset = offset + i;
			auto curOffset_uv = offset_uv + (i >> 1);
			Npp32u expectedValue_y = 0;
			Npp32u expectedValue_u = 0;
			Npp32u expectedValue_v = 0;
			RGB_TO_YUV(h_R8u[curOffset], h_G8u[curOffset], h_B8u[curOffset], expectedValue_y, expectedValue_u, expectedValue_v);

			auto actualValue_y = static_cast<Npp32u>(h_y8u[curOffset]);
			auto actualValue_u = -1;
			auto actualValue_v = -1;
			COMPARE_AND_PRINT(expectedValue_y, actualValue_y, 1, equal, std::to_string(j) + "<>" + std::to_string(i) + "<y>");

			if (i % 2 == 0 && j % 2 == 0)
			{
				actualValue_u = static_cast<Npp32u>(h_u8u[curOffset_uv]);
				actualValue_v = static_cast<Npp32u>(h_v8u[curOffset_uv]);
				COMPARE_AND_PRINT(expectedValue_u, actualValue_u, 1, equal, std::to_string(j) + "<>" + std::to_string(i) + "<u>");
				COMPARE_AND_PRINT(expectedValue_v, actualValue_v, 1, equal, std::to_string(j) + "<>" + std::to_string(i) + "<v>");
			}
			if (!equal)
			{
				std::cout << j << "<>" << i << "<input>" << static_cast<Npp32u>(h_R8u[curOffset]) << "<>" << static_cast<Npp32u>(h_G8u[curOffset]) << "<>" << static_cast<Npp32u>(h_B8u[curOffset]) << "<output>" << actualValue_y << "<>" << actualValue_u << "<>" << actualValue_v << std::endl;
			}
		}
		if (!equal)
		{
			throw "failed";
		}
	}

	ck(cudaStreamDestroy(stream));
}

void testHSVToRGB_randomvalues(int argc, char **argv)
{
	std::string method = "";
	if (argc == 1)
	{
		method = argv[0];
	}

	DeviceBuffer r, g, b, R, G, B;

	int width = 1920;
	int height = 1080;

	r.init(width, height);
	g.init(width, height);
	b.init(width, height);
	R.init(width, height);
	G.init(width, height);
	B.init(width, height);

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


	HostBuffer h_r, h_g, h_b, h_R, h_G, h_B;
	h_r.init(width, height);
	h_g.init(width, height);
	h_b.init(width, height);
	h_R.init(width, height);
	h_G.init(width, height);
	h_B.init(width, height);
	h_r.setAllValues(40);
	h_g.setAllValues(130);
	h_b.setAllValues(200);
	h_r.copyTo(r);
	h_g.copyTo(g);
	h_b.copyTo(b);

	R.memset(40);
	G.memset(130);
	B.memset(200);

	float hue = 0.1;
	float saturation = 2;
	launch_rgbhuesaturation(r8u, g8u, b8u, R8u, G8u, B8u, hue, saturation, step, size, stream, "");
	ck(cudaStreamSynchronize(stream));

	h_R.copy(R);
	h_G.copy(G);
	h_B.copy(B);

	auto h_r8u = static_cast<uint8_t *>(h_r.data());
	auto h_g8u = static_cast<uint8_t *>(h_g.data());
	auto h_b8u = static_cast<uint8_t *>(h_b.data());
	auto h_R8u = static_cast<uint8_t *>(h_R.data());
	auto h_G8u = static_cast<uint8_t *>(h_G.data());
	auto h_B8u = static_cast<uint8_t *>(h_B.data());

	bool equal = true;
	for (auto j = 0; j < height && equal; j++)
	{
		auto offset = width * j;
		for (auto i = 0; i < width && equal; i++)
		{
			auto curOffset = offset + i;
			Npp32u expectedValue_R = 0;
			Npp32u expectedValue_G = 0;
			Npp32u expectedValue_B = 0;
			RGBHUESATURATIONADJUST(h_r8u[curOffset], h_g8u[curOffset], h_b8u[curOffset], expectedValue_R, expectedValue_G, expectedValue_B, hue, saturation);

			auto actualValue_R = static_cast<Npp32u>(h_R8u[curOffset]);
			auto actualValue_G = static_cast<Npp32u>(h_G8u[curOffset]);
			auto actualValue_B = static_cast<Npp32u>(h_B8u[curOffset]);

			int tolerance = 0;

			bool equal_r = true;
			bool equal_g = true;
			bool equal_b = true;
			COMPARE_AND_PRINT(expectedValue_R, actualValue_R, tolerance, equal_r, std::to_string(j) + "<>" + std::to_string(i) + "<r>");
			COMPARE_AND_PRINT(expectedValue_G, actualValue_G, tolerance, equal_g, std::to_string(j) + "<>" + std::to_string(i) + "<g>");
			COMPARE_AND_PRINT(expectedValue_B, actualValue_B, tolerance, equal_b, std::to_string(j) + "<>" + std::to_string(i) + "<b>");

			equal = equal_r && equal_g && equal_b;
			if (!equal)
			{
				std::cout << j << "<>" << i << "<input>" << static_cast<Npp32u>(h_r8u[curOffset]) << "<>" << static_cast<Npp32u>(h_g8u[curOffset]) << "<>" << static_cast<Npp32u>(h_b8u[curOffset]) << "<output>" << actualValue_R << "<>" << actualValue_G << "<>" << actualValue_B << std::endl;
			}
		}
		if (!equal)
		{
			throw "failed";
		}
	}



	ck(cudaStreamDestroy(stream));
}


void testYUV420HueSaturation_randomvalues(int argc, char **argv)
{
	testYUV42ToRGB_randomvalues(argc, argv);
	testHSVToRGB_randomvalues(argc, argv);
	//return;

	std::string method = "";
	if (argc == 1)
	{
		method = argv[0];
	}

	DeviceBuffer y, u, v, Y, U, V;

	int width = 1920;
	int height = 1080;

	int width_2 = width >> 1;
	int height_2 = height >> 1;

	y.init(width, height);
	u.init(width_2, height_2);
	v.init(width_2, height_2);
	Y.init(width, height);
	U.init(width_2, height_2);
	V.init(width_2, height_2);

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


	HostBuffer h_y, h_u, h_v, h_Y, h_U, h_V;
	h_y.init(width, height);
	h_u.init(width_2, height_2);
	h_v.init(width_2, height_2);
	h_Y.init(width, height);
	h_U.init(width_2, height_2);
	h_V.init(width_2, height_2);
	h_y.setAllValues(5);
	h_u.setAllValues(20);
	h_v.setAllValues(190);
	h_y.copyTo(y);
	h_u.copyTo(u);
	h_v.copyTo(v);


	float hue = 0.2;
	float saturation = 2;
	float brightness = 20;
	float contrast = 0.2;

	Y.memset(0);
	U.memset(0);
	V.memset(0);


	launch_yuv420huesaturation(y8u, u8u, v8u, Y8u, U8u, V8u, brightness, contrast, hue, saturation, step_y, step_uv, size, stream, method);
	ck(cudaStreamSynchronize(stream));

	h_Y.copy(Y);
	h_U.copy(U);
	h_V.copy(V);

	auto h_y8u = static_cast<uint8_t *>(h_y.data());
	auto h_u8u = static_cast<uint8_t *>(h_u.data());
	auto h_v8u = static_cast<uint8_t *>(h_v.data());
	auto h_Y8u = static_cast<uint8_t *>(h_Y.data());
	auto h_U8u = static_cast<uint8_t *>(h_U.data());
	auto h_V8u = static_cast<uint8_t *>(h_V.data());

	bool equal = true;
	for (auto j = 0; j < height && equal; j++)
	{
		auto offset = width * j;
		int offset_uv = width_2 * (j >> 1);
		for (auto i = 0; i < width && equal; i++)
		{
			auto curOffset = offset + i;
			auto curOffset_uv = offset_uv + (i >> 1);
			Npp32u expectedValue_y = 0;
			Npp32u expectedValue_u = 0;
			Npp32u expectedValue_v = 0;
			YUVHUESATURATIONADJUST(h_y8u[curOffset], h_u8u[curOffset_uv], h_v8u[curOffset_uv], expectedValue_y, expectedValue_u, expectedValue_v, brightness, contrast, hue, saturation);

			auto actualValue_y = static_cast<Npp32u>(h_Y8u[curOffset]);
			auto actualValue_u = -1;
			auto actualValue_v = -1;
			bool equal_y = true;
			bool equal_u = true;
			bool equal_v = true;
			int tolerance = 1;
			COMPARE_AND_PRINT(expectedValue_y, actualValue_y, tolerance, equal_y, std::to_string(j) + "<>" + std::to_string(i) + "<y>");

			if (i % 2 == 0 && j % 2 == 0)
			{
				actualValue_u = static_cast<Npp32u>(h_U8u[curOffset_uv]);
				actualValue_v = static_cast<Npp32u>(h_V8u[curOffset_uv]);
				COMPARE_AND_PRINT(expectedValue_u, actualValue_u, tolerance, equal_u, std::to_string(j) + "<>" + std::to_string(i) + "<u>");
				COMPARE_AND_PRINT(expectedValue_v, actualValue_v, tolerance, equal_v, std::to_string(j) + "<>" + std::to_string(i) + "<v>");
			}

			equal = equal_y && equal_u && equal_v;
			if (!equal)
			{
				std::cout << j << "<>" << i << "<input>" << static_cast<Npp32u>(h_y8u[curOffset]) << "<>" << static_cast<Npp32u>(h_u8u[curOffset_uv]) << "<>" << static_cast<Npp32u>(h_v8u[curOffset_uv]) << "<output>" << actualValue_y << "<>" << actualValue_u << "<>" << actualValue_v << std::endl;
			}

		}
		if (!equal)
		{
			throw "failed";
		}
	}
	std::cout << "test values done" << std::endl;

	ck(cudaStreamDestroy(stream));
}

