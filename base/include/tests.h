#pragma once

#include <thread>
#include <cuda_runtime_api.h>
#include <iostream>
#include "Buffer.h"

// tests
void testAdd(int argc, char **argv);
void testAddNPP();
void testAddC(int argc, char **argv);
void testAddCNPP();
void testAddCMulC(int argc, char **argv);
void testBrightnessContrast_uv_int8(int argc, char **argv);
void testAddCMulCNPP();
void testMulC(int argc, char **argv);
void testMulCNPP();


// color conversion tests
void testYUV420ToRGB(int argc, char **argv);
void testYUV420ToRGBNPP();
void testRGBToYUV420(int argc, char **argv);
void testRGBToYUV420NPP();

void testRGBToHSVNPP();
void testRGBHueSaturation(int argc, char **argv);
void testYUV420HueSaturation(int argc, char **argv);

void getDeviceBuffer(int width, int height, int value, DeviceBuffer& buffer);
void profile(std::function<void()> compute);
bool copyAndCheckValue(DeviceBuffer& buffer, int value);