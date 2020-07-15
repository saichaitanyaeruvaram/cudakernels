#include "tests.h"
#include "cuda_runtime.h"
#include "common.h"


void getDeviceBuffer(int width, int height, int value, DeviceBuffer& buffer)
{
    buffer.init(width, height);
	buffer.memset(value);    
}

bool copyAndCheckValue(DeviceBuffer& buffer, int value)
{
	HostBuffer hostBuffer;
	hostBuffer.init(buffer);
	hostBuffer.copy(buffer);

	auto width = buffer.width();
	auto height = buffer.height();
	auto data = static_cast<uint8_t*>(hostBuffer.data());

	bool equal = true;
	for (auto i = 0; i < height; i++)
	{
		auto offset = width * i;
		for (auto j = 0; j < width && equal; j++)
		{
			auto actualValue = static_cast<int>(data[offset + j]);
			if (actualValue != value)
			{
				std::cout << "row<" << i << "> col<" << j <<  "> expected<" << value << "> actual<" << actualValue << ">" << std::endl;
				equal = false;
				break;
			}
		}
	}


	if (!equal)
	{
		throw "copyAndCheckValue failed";
	}
	
	return true;
}