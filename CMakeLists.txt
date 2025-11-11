cmake_minimum_required(VERSION 3.18)

project(cuda-khun-poker LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 75 86)

file(GLOB CUDA_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cu)

file(GLOB CUDA_HEADERS ${CMAKE_SOURCE_DIR}/include/*.cuh)

add_executable(cuda-khun-poker ${CUDA_SOURCES} ${CUDA_HEADERS})

target_include_directories(cuda-khun-poker PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(WIN32)
    source_group("Source Files" FILES ${CUDA_SOURCES})
    source_group("Headers Files" FILES ${CUDA_HEADERS})
endif()

set(CUDA_ARCHITECTURES 75 80 86 89)
