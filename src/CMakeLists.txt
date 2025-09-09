cmake_minimum_required(VERSION 3.23)

project(gpt2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG_ROOT environment variable is not set. Please set it to the root directory of your vcpkg installation.")
endif()

# Use the VCPKG_ROOT environment variable to set the CMAKE_TOOLCHAIN_FILE
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(absl_DIR "$ENV{VCPKG_ROOT}/installed/x64-mingw-static/share/absl")
set(re2_DIR "$ENV{VCPKG_ROOT}/installed/x64-mingw-static/share/re2")
find_package(re2 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} gpt2.cpp bpe_tokenizer.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE re2::re2)

file(COPY "${CMAKE_SOURCE_DIR}/assets/gpt2-small-124M.bin" DESTINATION "${CMAKE_BINARY_DIR}/src")
file(COPY "${CMAKE_SOURCE_DIR}/assets/vocab.json" DESTINATION "${CMAKE_BINARY_DIR}/src")
file(COPY "${CMAKE_SOURCE_DIR}/assets/merges.txt" DESTINATION "${CMAKE_BINARY_DIR}/src")