#include "tests.h"
#include "cuda_runtime.h"
#include "common.h"


void getDeviceBuffer(int width, int height, int value, DeviceBuffer& buffer, int& step)
{
    buffer.init(width, height);
        
    ck(cudaMemset(buffer.data(), value, buffer.size()));
    ck(cudaDeviceSynchronize());
}

bool copyAndCheckValue(DeviceBuffer& buffer, int value)
{
	HostBuffer hostBuffer;
	hostBuffer.init(buffer);
	hostBuffer.copy(buffer);

	auto width = buffer.width();
	auto height = buffer.height();
	auto data = static_cast<uint8_t*>(hostBuffer.data());
	for (auto i = 0; i < height; i++)
	{
		auto offset = width * i;
		for (auto j = 0; j < width; j++)
		{
			if (data[offset + j] != value)
			{
				return false;
			}
		}
	}

	return true;
}