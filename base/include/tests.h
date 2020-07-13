#pragma once

#include <thread>
#include <cuda_runtime_api.h>
#include <iostream>
#include "Buffer.h"

// tests
void testAdd();
void testAddNPP();

void getDeviceBuffer(int width, int height, int value, DeviceBuffer& buffer, int& step);
void profile(std::function<void()> compute);
bool copyAndCheckValue(DeviceBuffer& buffer, int value);