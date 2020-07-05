#######################################################################
# CMake build script for cnplugin

# name of the project
# description of the project
message(STATUS "[cnplugin] " "build cnplugin")

project(cnplugin)

# least support cmake version
# in case we need other new policy feathers
cmake_minimum_required(VERSION 2.8)

# set CMAKE_C_FLAGS
# set CMAKE_CXX_FLAGS
# etc..

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

set(NEUWARE $ENV{NEUWARE})

#######################################################################
# -- Release and Debug for global control
# -- Note that add_definition will pass definition to subdirectories like
#    runtime_api and NGPF

if(${CMAKE_BUILD_TYPE} MATCHES "release")
  message("RELEASE")
else()
  message("DEBUG")
endif()

#
#######################################################################
#
# -- compile and link
set(USE_CPP_LIB stdc++)

# WARNING: Add top level include directories with CAUTION!!!!!!
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/common/include")
include_directories("${NEUWARE}/include")

# output path
link_directories("${CMAKE_BINARY_DIR}/lib")

message("Building ${VENDOR_NAME}")

