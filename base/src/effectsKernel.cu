#include "kernels.h"

#define BLOCK_SIZE 1024
#define UINT32_BLOCK_STEP_2 64 // 16X4
#define BLOCK_SIZE_4 256
//
//__global__ void rgbtohsvtorgb(const Npp8u* R, const Npp8u* G, const Npp8u* B, Npp8u* Rn, Npp8u* Bn, Npp8u* Gn)
//{
//	Npp32f nNormalizedR = (Npp32f)R * 0.003921569F; // / 255.0F
//	Npp32f nNormalizedG = (Npp32f)G * 0.003921569F;
//	Npp32f nNormalizedB = (Npp32f)B * 0.003921569F;
//	Npp32f nS;
//	Npp32f nH;
//	// Value
//	Npp32f nV = fmaxf(nNormalizedR, nNormalizedG);
//	nV = fmaxf(nV, nNormalizedB);
//	// Saturation
//	Npp32f nTemp = fminf(nNormalizedR, nNormalizedG);
//	nTemp = fminf(nTemp, nNormalizedB);
//	Npp32f nDivisor = nV - nTemp;
//	if (nV == 0.0F) // achromatics case
//	{
//		nS = 0.0F;
//		nH = 0.0F;
//	}
//	else // chromatics case
//		nS = nDivisor / nV;
//	// Hue:
//	Npp32f nCr = (nV - nNormalizedR) / nDivisor;
//	Npp32f nCg = (nV - nNormalizedG) / nDivisor;
//	Npp32f nCb = (nV - nNormalizedB) / nDivisor;
//	if (nNormalizedR == nV)
//		nH = nCb - nCg;
//	else if (nNormalizedG == nV)
//		nH = 2.0F + nCr - nCb;
//	else if (nNormalizedB == nV)
//		nH = 4.0F + nCg - nCr;
//	nH = nH * 0.166667F; // / 6.0F       
//	if (nH < 0.0F)
//		nH = nH + 1.0F;
//	H = (Npp8u)(nH * 255.0F);
//	S = (Npp8u)(nS * 255.0F);
//	V = (Npp8u)(nV * 255.0F);
//
//	Npp32f nNormalizedH = (Npp32f)H * 0.003921569F; // / 255.0F
//	Npp32f nNormalizedS = (Npp32f)S * 0.003921569F;
//	Npp32f nNormalizedV = (Npp32f)V * 0.003921569F;
//	Npp32f nR;
//	Npp32f nG;
//	Npp32f nB;
//	if (nNormalizedS == 0.0F)
//	{
//		nR = nG = nB = nNormalizedV;
//	}
//	else
//	{
//		if (nNormalizedH == 1.0F)
//			nNormalizedH = 0.0F;
//		else
//			nNormalizedH = nNormalizedH * 6.0F; // / 0.1667F
//	}
//	Npp32f nI = floorf(nNormalizedH);
//	Npp32f nF = nNormalizedH - nI;
//	Npp32f nM = nNormalizedV * (1.0F - nNormalizedS);
//	Npp32f nN = nNormalizedV * (1.0F - nNormalizedS * nF);
//	Npp32f nK = nNormalizedV * (1.0F - nNormalizedS * (1.0F - nF));
//	if (nI == 0.0F)
//	{
//		nR = nNormalizedV; nG = nK; nB = nM;
//	}
//	else if (nI == 1.0F)
//	{
//		nR = nN; nG = nNormalizedV; nB = nM;
//	}
//	else if (nI == 2.0F)
//	{
//		nR = nM; nG = nNormalizedV; nB = nK;
//	}
//	else if (nI == 3.0F)
//	{
//		nR = nM; nG = nN; nB = nNormalizedV;
//	}
//	else if (nI == 4.0F)
//	{
//		nR = nK; nG = nM; nB = nNormalizedV;
//	}
//	else if (nI == 5.0F)
//	{
//		nR = nNormalizedV; nG = nM; nB = nN;
//	}
//	R = (Npp8u)(nR * 255.0F);
//	G = (Npp8u)(nG * 255.0F);
//	B = (Npp8u)(nB * 255.0F);
//
//}
//


#define CLAMP_1(x) x < 0 ? 0 : (x > 1 ? 1 : x)
#define CLAMP_255(x) x < 0 ? 0 : (x > 255 ? 255 : x)
#define CLAMP_int8(x) x < -128 ? -128 : (x > 127 ? 127 : x)

// https://en.wikipedia.org/wiki/YUV#Y%E2%80%B2UV420p_(and_Y%E2%80%B2V12_or_YV12)_to_RGB888_conversion

//#define YUV_TO_RGB( Y, U, V, R, G, B )                      \
//    do                                                      \
//    {                                                       \
//		int rTmp = Y + (1.370705 * V);                      \
//        int gTmp = Y - (0.698001 * V) - (0.337633 * U);     \
//        int bTmp = Y + (1.732446 * U);						\
//        R = CLAMP_255(rTmp);                                \
//        G = CLAMP_255(gTmp);                                \
//        B = CLAMP_255(bTmp);                                \
//	} while (0)

#define YUV_TO_RGB( Y, U, V, R, G, B )											\
    do																			\
    {																			\
		float rTmp = Y + __fmul_rn (1.370705, V);								\
        float gTmp = Y - __fmul_rn (0.698001, V) - __fmul_rn (0.337633, U);     \
        float bTmp = Y + __fmul_rn (1.732446, U);								\
        R = CLAMP_255(rTmp);													\
        G = CLAMP_255(gTmp);													\
        B = CLAMP_255(bTmp);													\
	} while (0)

#define RGB_TO_Y( R, G, B, Y )															 \
    do																					 \
    {																					 \
		int yTmp = __fmul_rn(R, 0.299) + __fmul_rn (0.587, G) +  __fmul_rn (0.114, B); \
        Y = CLAMP_255(yTmp);															 \
    } while (0)

#define RGB_TO_UV( R, G, B, U, V )														 \
    do																					 \
    {																					 \
		int uTmp = __fmul_rn(B, 0.436) - __fmul_rn (0.289, G) -  __fmul_rn (0.147, R); \
        int vTmp = __fmul_rn(R, 0.615) - __fmul_rn (0.515, G) -  __fmul_rn (0.1, B);	 \
        U = 128 + (CLAMP_int8(uTmp));														 \
        V = 128 + (CLAMP_int8(vTmp));														 \
	} while (0)

#define RGB_TO_HSV(R, G, B, H, S, V) 									\
do  																	\
{ 																		\
	Npp32f nNormalizedR = __fmul_rn(R, 0.003921569F); /* 255.0F*/ 		\
	Npp32f nNormalizedG = __fmul_rn(G, 0.003921569F); 					\
	Npp32f nNormalizedB = __fmul_rn(B, 0.003921569F); 					\
	Npp32f nS; 															\
	Npp32f nH; 															\
	/* Value*/ 															\
	Npp32f nV = fmaxf(nNormalizedR, nNormalizedG); 						\
	nV = fmaxf(nV, nNormalizedB); 										\
	/*Saturation*/ 														\
	Npp32f nTemp = fminf(nNormalizedR, nNormalizedG); 					\
	nTemp = fminf(nTemp, nNormalizedB); 								\
	Npp32f nDivisor = nV - nTemp; 										\
	if (nV == 0.0F) /*achromatics case*/ 								\
	{ 																	\
		nS = 0.0F; 														\
		nH = 0.0F; 														\
	} 																	\
	else /*chromatics case*/ 											\
	{ 																	\
		nS = __fdiv_rn(nDivisor, nV); 									\
	} 																	\
	/* Hue:*/ 															\
	Npp32f nCr = __fdiv_rn((nV - nNormalizedR), nDivisor); 				\
	Npp32f nCg = __fdiv_rn((nV - nNormalizedG), nDivisor); 				\
	Npp32f nCb = __fdiv_rn((nV - nNormalizedB), nDivisor); 				\
	if (nNormalizedR == nV) 											\
		nH = nCb - nCg; 												\
	else if (nNormalizedG == nV) 										\
		nH = 2.0F + nCr - nCb; 											\
	else if (nNormalizedB == nV) 										\
		nH = 4.0F + nCg - nCr; 											\
	nH = __fmul_rn(nH, 0.166667F); /* 6.0F*/        					\
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
			nNormalizedH = __fmul_rn(nNormalizedH, 6.0F);  									\
		}																					\
	} 																						\
	Npp32f nI = floorf(nNormalizedH); 														\
	Npp32f nF = nNormalizedH - nI; 															\
	Npp32f nM = __fmul_rn(nNormalizedV, (1.0F - nNormalizedS)); 							\
	Npp32f nN = __fmul_rn(nNormalizedV, (1.0F - __fmul_rn(nNormalizedS, nF) )	); 			\
	Npp32f nK = __fmul_rn(nNormalizedV, (1.0F - __fmul_rn(nNormalizedS, (1.0F - nF)) ) ); 	\
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
	R = CLAMP_255(__fmul_rn(nR, 255.0F)); 													\
	G = CLAMP_255(__fmul_rn(nG, 255.0F)); 													\
	B = CLAMP_255(__fmul_rn(nB, 255.0F)); 													\
																							\
} while(0)

#define RGBHUESATURATIONADJUST(r, g, b, R, G, B, hue, saturation)   \
do 																	\
{ 																	\
	Npp32f H, S, V; 												\
	RGB_TO_HSV(r, g, b, H, S, V); 									\
	H = CLAMP_1(H + hue); 											\
	S = CLAMP_1(__fmul_rn(S, saturation)); 							\
	HSV_TO_RGB(H, S, V, R, G, B);									\
} while(0)

#define YUVHUESATURATIONADJUST_Y(y, u, v, Y, hue, saturation) 		\
do 																	\
{ 																	\
	Npp8u r, g, b; 													\
	YUV_TO_RGB(y, u, v, r, g, b); 									\
	Npp8u R, G, B; 													\
	RGBHUESATURATIONADJUST(r, g, b, R, G, B, hue, saturation); 		\
	RGB_TO_Y(R, G, B, Y); 											\
} while(0)

#define YUVHUESATURATIONADJUST(y, u, v, Y, U, V, hue, saturation) 	\
do 																	\
{ 																	\
	Npp8u r, g, b; 													\
	YUV_TO_RGB(y, u, v, r, g, b); 									\
	Npp8u R, G, B; 													\
	RGBHUESATURATIONADJUST(r, g, b, R, G, B, hue, saturation); 		\
	RGB_TO_Y(R, G, B, Y); 											\
	RGB_TO_UV(R, G, B, U, V); 										\
} while (0)

__global__ void yuv420torgb(const uchar4* Y, const uint32_t* U, const uint32_t* V, uchar4* r, uchar4* g, uchar4* b, int width, int height, int step_y, int step_uv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step_y + x;

	__shared__ uint32_t u_data[BLOCK_SIZE_4];
	__shared__ uint32_t v_data[BLOCK_SIZE_4];

	// read for every 4 frames once            
	if (threadIdx.x % 2 == 0 && threadIdx.y % 2 == 0)
	{
		// 16*threadIdx.y*0.5 + threadIdx.x*0.5
		auto uvThreadOffset = (threadIdx.y << 3) + (threadIdx.x >> 1);
		auto uvOffset = (y >> 1) * (step_uv)+(x >> 1);
		u_data[uvThreadOffset] = U[uvOffset];
		v_data[uvThreadOffset] = V[uvOffset];
	}

	__syncthreads();

	// 32x32x4 y, r, g, b values
	// 16x16x4 u, v values

	auto u_data_uint8 = reinterpret_cast<uint8_t*>(u_data);
	auto v_data_uint8 = reinterpret_cast<uint8_t*>(v_data);

	auto uvThreadOffset = (threadIdx.y >> 1)*UINT32_BLOCK_STEP_2 + (threadIdx.x << 1);
	int u_value = u_data_uint8[uvThreadOffset] - 128;
	int v_value = v_data_uint8[uvThreadOffset] - 128;

	YUV_TO_RGB(Y[offset].x, u_value, v_value, r[offset].x, g[offset].x, b[offset].x);
	YUV_TO_RGB(Y[offset].y, u_value, v_value, r[offset].y, g[offset].y, b[offset].y);

	uvThreadOffset += 1;
	u_value = u_data_uint8[uvThreadOffset] - 128;
	v_value = v_data_uint8[uvThreadOffset] - 128;
	YUV_TO_RGB(Y[offset].z, u_value, v_value, r[offset].z, g[offset].z, b[offset].z);
	YUV_TO_RGB(Y[offset].w, u_value, v_value, r[offset].w, g[offset].w, b[offset].w);
}

__global__ void yuv420torgb_plain2(const uint8_t* Y, const uint8_t* U, const uint8_t* V, uint8_t* r, uint8_t* g, uint8_t* b, int width, int height, int step_y, int step_uv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step_y + x;
	auto uvOffset = (y >> 1) * (step_uv)+(x >> 1);

	int u_value = U[uvOffset] - 128;
	int v_value = V[uvOffset] - 128;

	YUV_TO_RGB(Y[offset], u_value, v_value, r[offset], g[offset], b[offset]);
}

__global__ void yuv420torgb_plain(const uchar4* Y, const uint8_t* U, const uint8_t* V, uchar4* r, uchar4* g, uchar4* b, int width, int height, int step_y, int step_uv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step_y + x;
	auto uvOffset = (y >> 1) * (step_uv)+(x << 1);

	int u_value = U[uvOffset] - 128;
	int v_value = V[uvOffset] - 128;

	YUV_TO_RGB(Y[offset].x, u_value, v_value, r[offset].x, g[offset].x, b[offset].x);
	YUV_TO_RGB(Y[offset].y, u_value, v_value, r[offset].y, g[offset].y, b[offset].y);

	uvOffset += 1;
	u_value = U[uvOffset] - 128;
	v_value = V[uvOffset] - 128;
	YUV_TO_RGB(Y[offset].z, u_value, v_value, r[offset].z, g[offset].z, b[offset].z);
	YUV_TO_RGB(Y[offset].w, u_value, v_value, r[offset].w, g[offset].w, b[offset].w);
}

__global__ void rgbtoyuv420(const uchar4* R, const uchar4* G, const uchar4* B, uchar4* Y, uint8_t* U, uint8_t* V, int width, int height, int step_y, int step_uv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step_y + x;
	RGB_TO_Y(R[offset].x, G[offset].x, B[offset].x, Y[offset].x);
	RGB_TO_Y(R[offset].y, G[offset].y, B[offset].y, Y[offset].y);
	RGB_TO_Y(R[offset].z, G[offset].z, B[offset].z, Y[offset].z);
	RGB_TO_Y(R[offset].w, G[offset].w, B[offset].w, Y[offset].w);

	if (y % 2 == 0)
	{
		auto uvOffset = (y >> 1) * (step_uv)+(x << 1);
		RGB_TO_UV(R[offset].x, G[offset].x, B[offset].x, U[uvOffset], V[uvOffset]);
		uvOffset += 1;
		RGB_TO_UV(R[offset].z, G[offset].z, B[offset].z, U[uvOffset], V[uvOffset]);
	}

}

__global__ void rgbhuesaturation(const uchar4* r, const uchar4* g, const uchar4* b, uchar4* R, uchar4* G, uchar4* B, Npp32f hue, Npp32f saturation, int width, int height, int step)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step + x;
	RGBHUESATURATIONADJUST(r[offset].x, g[offset].x, b[offset].x, R[offset].x, G[offset].x, B[offset].x, hue, saturation);
	RGBHUESATURATIONADJUST(r[offset].y, g[offset].y, b[offset].y, R[offset].y, G[offset].y, B[offset].y, hue, saturation);
	RGBHUESATURATIONADJUST(r[offset].z, g[offset].z, b[offset].z, R[offset].z, G[offset].z, B[offset].z, hue, saturation);
	RGBHUESATURATIONADJUST(r[offset].w, g[offset].w, b[offset].w, R[offset].w, G[offset].w, B[offset].w, hue, saturation);
}

__global__ void yuv420huesaturation(const uchar4* Yold, const uint8_t* Uold, const uint8_t* Vold, uchar4* Y, uint8_t* U, uint8_t* V, Npp32f hue, Npp32f saturation, int width, int height, int step_y, int step_uv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = y * step_y + x;
	auto uvOffset = (y >> 1) * (step_uv)+(x << 1);

	int u_value = Uold[uvOffset] - 128;
	int v_value = Vold[uvOffset] - 128;

	if (y % 2 == 0)
	{
		YUVHUESATURATIONADJUST(Yold[offset].x, u_value, v_value, Y[offset].x, U[uvOffset], V[uvOffset], hue, saturation);		
	}
	else
	{
		YUVHUESATURATIONADJUST_Y(Yold[offset].x, u_value, v_value, Y[offset].x, hue, saturation);
	}
	YUVHUESATURATIONADJUST_Y(Yold[offset].y, u_value, v_value, Y[offset].y, hue, saturation);

	uvOffset += 1;
	u_value = U[uvOffset] - 128;
	v_value = V[uvOffset] - 128;
	if (y % 2 == 0)
	{		
		YUVHUESATURATIONADJUST(Yold[offset].z, u_value, v_value, Y[offset].z, U[uvOffset], V[uvOffset], hue, saturation);
	}
	else
	{
		YUVHUESATURATIONADJUST_Y(Yold[offset].z, u_value, v_value, Y[offset].z, hue, saturation);
	}
	YUVHUESATURATIONADJUST_Y(Yold[offset].w, u_value, v_value, Y[offset].w, hue, saturation);
}

void launch_yuv420torgb(const Npp8u* Y, const Npp8u* U, const Npp8u* V, Npp8u* R, Npp8u* G, Npp8u* B, int step_y, int step_uv, NppiSize size, cudaStream_t stream, std::string method)
{
	if (method == "plain")
	{
		auto width = size.width >> 2;
		step_y = step_y >> 2;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		yuv420torgb_plain << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(Y), reinterpret_cast<const uint8_t*>(U), reinterpret_cast<const uint8_t*>(V), reinterpret_cast<uchar4*>(R), reinterpret_cast<uchar4*>(G), reinterpret_cast<uchar4*>(B), width, size.height, step_y, step_uv);
	}
	else if (method == "plain2")
	{
		auto width = size.width;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		yuv420torgb_plain2 << <grid, block, 0, stream >> > (reinterpret_cast<const uint8_t*>(Y), reinterpret_cast<const uint8_t*>(U), reinterpret_cast<const uint8_t*>(V), reinterpret_cast<uint8_t*>(R), reinterpret_cast<uint8_t*>(G), reinterpret_cast<uint8_t*>(B), size.width, size.height, step_y, step_uv);
	}
	else
	{
		auto width = size.width >> 2;
		step_y = step_y >> 2;
		step_uv = step_uv >> 2;
		dim3 block(32, 32);
		dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
		yuv420torgb << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(Y), reinterpret_cast<const uint32_t*>(U), reinterpret_cast<const uint32_t*>(V), reinterpret_cast<uchar4*>(R), reinterpret_cast<uchar4*>(G), reinterpret_cast<uchar4*>(B), width, size.height, step_y, step_uv);
	}
}

void launch_rgbtoyuv420(const Npp8u* R, const Npp8u* G, const Npp8u* B, Npp8u* Y, Npp8u* U, Npp8u* V, int step_y, int step_uv, NppiSize size, cudaStream_t stream, std::string method)
{
	auto width = size.width >> 2;
	step_y = step_y >> 2;
	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
	rgbtoyuv420 << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(R), reinterpret_cast<const uchar4*>(G), reinterpret_cast<const uchar4*>(B), reinterpret_cast<uchar4*>(Y), reinterpret_cast<uint8_t*>(U), reinterpret_cast<uint8_t*>(V), width, size.height, step_y, step_uv);

}

void launch_rgbtohsv(const Npp8u* R, const Npp8u* G, const Npp8u* B, Npp8u* H, Npp8u* S, Npp8u* V, int step, NppiSize size, cudaStream_t stream, std::string method)
{
	auto width = size.width >> 2;
	step = step >> 2;
	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
	// rgbtohsv << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(R), reinterpret_cast<const uchar4*>(G), reinterpret_cast<const uchar4*>(B), reinterpret_cast<uchar4*>(H), reinterpret_cast<uchar4*>(S), reinterpret_cast<uchar4*>(V),  width, size.height, step);
}

void launch_rgbhuesaturation(const Npp8u* r, const Npp8u* g, const Npp8u* b, Npp8u* R, Npp8u* G, Npp8u* B, Npp32f hue, Npp32f saturation, int step, NppiSize size, cudaStream_t stream, std::string method)
{
	auto width = size.width >> 2;
	step = step >> 2;
	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
	rgbhuesaturation << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(r), reinterpret_cast<const uchar4*>(g), reinterpret_cast<const uchar4*>(b), reinterpret_cast<uchar4*>(R), reinterpret_cast<uchar4*>(G), reinterpret_cast<uchar4*>(B), hue, saturation, width, size.height, step);
}

void launch_yuv420huesaturation(const Npp8u* y, const Npp8u* u, const Npp8u* v, Npp8u* Y, Npp8u* U, Npp8u* V, Npp32f hue, Npp32f saturation, int step_y, int step_uv, NppiSize size, cudaStream_t stream, std::string method)
{
	auto width = size.width >> 2;
	step_y = step_y >> 2;
	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (size.height + block.y - 1) / block.y);
	yuv420huesaturation << <grid, block, 0, stream >> > (reinterpret_cast<const uchar4*>(y), reinterpret_cast<const uint8_t*>(u), reinterpret_cast<const uint8_t*>(v), reinterpret_cast<uchar4*>(Y), reinterpret_cast<uint8_t*>(U), reinterpret_cast<uint8_t*>(V), hue, saturation, width, size.height, step_y, step_uv);
}