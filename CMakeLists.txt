cmake_minimum_required(VERSION 3.23)
project(testCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(testCuda main.cu)

set_target_properties(testCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
