cmake_minimum_required(VERSION 3.25)

project(cuda-khun-poker LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 75 86)

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/include/*.h ${PROJECT_SOURCE_DIR}/include/*.cuh)

add_executable(cuda-khun-poker ${SOURCES} ${HEADERS})

target_include_directories(cuda-khun-poker PRIVATE ${PROJECT_SOURCE_DIR}/include)

if(WIN32)
    source_group("Source Files" FILES ${SOURCES})
    source_group("Header Files" FILES ${HEADERS})
endif()
