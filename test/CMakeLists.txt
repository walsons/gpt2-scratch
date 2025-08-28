cmake_minimum_required(VERSION 3.23)

project(gpt2_tester)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_executable(${PROJECT_NAME} test.cpp)

file(COPY "${CMAKE_SOURCE_DIR}/assets/gpt2-small-124M.bin" DESTINATION "${CMAKE_BINARY_DIR}/test")
file(COPY "${CMAKE_SOURCE_DIR}/assets/vocab.json" DESTINATION "${CMAKE_BINARY_DIR}/test")
file(COPY "${CMAKE_SOURCE_DIR}/assets/merges.txt" DESTINATION "${CMAKE_BINARY_DIR}/test")