cmake_minimum_required(VERSION 3.18)

project(CUDATEST LANGUAGES CXX CUDA)
find_package(CUDAToolkit 10.1 REQUIRED EXACT) 
SET(NVCUDAToolkit_LIBS		
	cudart_static.lib  
	cuda.lib
	nvjpeg.lib
	nppig.lib
	nppicc.lib
	nppidei.lib
	nppial.lib
)

SET(SOURCE
	src/add.cpp
	src/buffers.cpp
	src/profile.cpp
	src/addkernel.cu
	src/main.cpp
)

SET(HEADERS
	include/common.h
	include/tests.h	
	include/kernels.h
)

include_directories( AFTER SYSTEM include)

add_executable(cudatest 
	${SOURCE}
	${HEADERS}
)

set_property(TARGET cudatest PROPERTY CUDA_ARCHITECTURES 30 32 52 53 60 61 62 70 72 75) 

target_link_libraries(cudatest   
  ${NVCUDAToolkit_LIBS}
)