cmake_minimum_required(VERSION 3.18)
project(NeuralNetworkLanguages CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

set(SOURCES
    src/Neural_Network.h
    src/Neural_Network.cu
    src/main.cpp
)

add_executable(prog ${SOURCES})

target_link_libraries(prog PRIVATE CUDA::cudart)
