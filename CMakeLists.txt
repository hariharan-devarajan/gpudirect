cmake_minimum_required(VERSION 3.14)
project(gpudirect LANGUAGES CUDA CXX)

if(NOT DEFINED ENV{pfs})
  message( FATAL_ERROR "Environment variable pfs is not defined. Exiting Cmake." )
endif()
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -rdynamic")
find_package(CUDA REQUIRED)
message("-- CUDA Found at ${CUDA_TOOLKIT_ROOT_DIR}")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CUDA_TOOLKIT_ROOT_DIR}/lib64")

set(GPD_PUBLIC_INCLUDE include/gpd/gpd.h)
set(GPD_PRIVATE_INCLUDE "")
set(GPD_SRC src/gpd/gpd.cpp)

add_library(gpd SHARED ${GPD_PUBLIC_INCLUDE} ${GPD_PRIVATE_INCLUDE} ${GPD_SRC})

#set(CMAKE_CXX_COMPILER ${CUDA_NVCC_EXECUTABLE})
add_executable(gpudirect main.cpp)
add_dependencies(gpudirect gpd)
target_link_libraries(gpudirect ${CUDA_LIBRARIES} -lcufile gpd -ldl -fPIC -rdynamic)

option(BUILD_TEST "Build the unit tests" ON)
if (BUILD_TEST)
    enable_testing()
endif ()

set(tss 1 4 64 1024 16384)
set(nos 1 10 100)
foreach(no ${nos})
    foreach(ts ${tss})
        add_test(NAME posix_${ts}_${no} COMMAND "${CMAKE_BINARY_DIR}/gpudirect" 0 $ENV{pfs}/gpudirect ${ts} ${no})
        set_property(TEST posix_${ts}_${no} APPEND PROPERTY ENVIRONMENT LD_PRELOAD=${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufile.so)
        add_test(NAME gpu_${ts}_${no} COMMAND "${CMAKE_BINARY_DIR}/gpudirect" 1 $ENV{pfs}/gpudirect ${ts} ${no})
        set_property(TEST gpu_${ts}_${no} APPEND PROPERTY ENVIRONMENT LD_PRELOAD=${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufile.so)
    endforeach()
endforeach()


function(print_directory_variables dir)
    # Dump variables:
    get_property(_variableNames DIRECTORY ${dir} PROPERTY VARIABLES)
    list (SORT _variableNames)
    foreach (_variableName ${_variableNames})
        get_directory_property(_variableValue DIRECTORY ${dir} DEFINITION ${_variableName})
        message(STATUS "DIR ${dir}: ${_variableName}=${_variableValue}")
    endforeach()
endfunction(print_directory_variables)

# for example
#print_directory_variables(.)
