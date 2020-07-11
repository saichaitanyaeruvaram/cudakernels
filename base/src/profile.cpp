#include "common.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using sys_clock = std::chrono::system_clock;

void profile(std::function<void()> compute)
{
    std::cout << "Starting warmup loops" << std::endl;
    for(auto i = 0; i < 10; i++)
    {
        compute();
    }
    std::cout << "Warmup loops done. Starting profiling" << std::endl;

	double totalTime = 0;
    for(auto i = 0; i < 10; i++)
    {
        auto start = sys_clock::now();
        for(auto j = 0; j < 1000; j++)
        {
            compute();
        }

        auto end = sys_clock::now();
        auto diff = end - start;
        totalTime += diff.count();

        auto timeElapsed = diff.count()/1000000000.0;
        double fps = 1000.0/timeElapsed;

        std::cout << "Processed 1000. Time Elapsed<" << timeElapsed << "> fps<" << static_cast<int>(fps) << ">" << std::endl;
    }

    totalTime = totalTime/1000000000.0;
    auto avgTime = totalTime/(10);
    double fps = 10*1000.0/(totalTime);

    std::cout << "AvgTime<" << avgTime << "> fps<" << static_cast<int>(fps) << ">" << std::endl;
}