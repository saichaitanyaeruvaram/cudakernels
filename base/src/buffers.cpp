#include "tests.h"
#include "cuda_runtime.h"
#include "common.h"

int getStep(int width)
{
    auto rem = width % 512;
    if (rem == 0)
    {
        return 0;
    }

    return width + (512 - rem);
}

int getSize(int width, int height, int& step)
{
    step = getStep(width);

    return step*height;
}

void getDeviceBuffer(int width, int height, int value, DeviceBuffer& buffer, int& step)
{
    auto size = getSize(width, height, step);

    buffer.init(size);
        
    ck(cudaMemset(buffer.data(), value, size));
    ck(cudaDeviceSynchronize());
}