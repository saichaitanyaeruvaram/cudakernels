#pragma once
#include "cuda_runtime.h"
#include "common.h"

class DeviceBuffer
{
public:
    DeviceBuffer(): ptr(nullptr)
    {

    }

    DeviceBuffer(size_t size)
    {
        init(size);
    }

    ~DeviceBuffer()
    {
		if (ptr)
		{
			ck(cudaFree(ptr));
		}
    }

    void init(size_t size)
    {
        ck(cudaMalloc(&ptr, size));
    }

    void *data()
    {
        return ptr;
    }

private:
    void *ptr;
};