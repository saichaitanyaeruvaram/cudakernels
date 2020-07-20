#pragma once
#include "cuda_runtime.h"
#include "common.h"

class Buffer
{
public:
	Buffer(size_t alignSize=0) : m_width(0),
		m_height(0),
		m_step(0),
		m_size(0),
		m_alignSize(alignSize),
		m_data(nullptr)
	{

	}

	virtual ~Buffer()
	{
		
	}

	virtual void init(int width, int height)
	{
		m_width = width;
		m_height = height;
		m_size = getSize(width, height, m_step);		
	}

	virtual void memset(int value)
	{

	}

	void *data()
	{
		return m_data;
	}

	size_t size()
	{
		return m_size;
	}

	int width()
	{
		return m_width;
	}

	int height()
	{
		return m_height;
	}

	int step()
	{
		return m_step;
	}

private:
	int getStep(int width)
	{
		if (m_alignSize == 0)
		{
			return width;
		}

		auto rem = width % m_alignSize;
		if (rem == 0)
		{
			return 0;
		}

		return width + (m_alignSize - rem);
	}

	int getSize(int width, int height, int& step)
	{
		step = getStep(width);

		return step * height;
	}

protected:
	void *m_data;
	size_t m_size;
	size_t m_alignSize;

	int m_width;
	int m_height;
	int m_step;
};

class DeviceBuffer: public Buffer
{
public:
	DeviceBuffer() : Buffer(512)
	{
		
	}
	
	~DeviceBuffer()
	{
		if (m_data)
		{
			ck(cudaFree(m_data));
		}
	}
	
	void init(int width, int height)
	{		
		Buffer::init(width, height);
		ck(cudaMalloc(&m_data, m_size));
	}	

	void memset(int value)
	{
		ck(cudaMemset(m_data, value, m_size));
		ck(cudaDeviceSynchronize());
	}
};

class HostBuffer: public Buffer
{
public:
	HostBuffer() : Buffer(0)
	{

	}

	~HostBuffer()
	{
		if (m_data)
		{
			delete[] m_data;
		}
	}

	void init(DeviceBuffer& buffer)
	{
		init(buffer.width(), buffer.height());
	}

	void init(int width, int height)
	{
		Buffer::init(width, height);
		m_data = static_cast<void*>(new uint8_t[m_size]);
	}

	void copy(DeviceBuffer& buffer)
	{
		ck(cudaMemcpy2D(m_data, m_step, buffer.data(), buffer.step(), buffer.width(), buffer.height(), cudaMemcpyDeviceToHost));
		ck(cudaDeviceSynchronize());
	}

	void copyTo(DeviceBuffer& buffer)
	{
		ck(cudaMemcpy2D(buffer.data(), buffer.step(), m_data, m_step, buffer.width(), buffer.height(), cudaMemcpyHostToDevice));
		ck(cudaDeviceSynchronize());
	}

	void setAllValues()
	{
		auto data = static_cast<uint8_t*>(m_data);

		int value = 0;
		for (auto i = 0; i < m_height; i++)
		{
			auto offset = m_width * i;
			for (auto j = 0; j < m_width; j++)
			{
				data[offset + j] = value++;
				if(value > 255)
				{
					value = 0;
				}
			}
		}
	}
};