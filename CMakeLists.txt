#  Cypress_example -- Example on how to use cypress as external project
#  Copyright (C) 2019 Christoph Ostrau
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

# CMake version and project metadata
cmake_minimum_required(VERSION 3.0)
project(BS2CYPRESS LANGUAGES CXX)

# Add the local "cmake" directory to the search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

# Enable C++14 support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC")

# Enable all warnings on MSVC and GCC/Clang/Intel
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
	endif()
elseif(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR
		("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
		("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

# Set a default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

find_package(PythonLibs 2.7 REQUIRED )

# Include the Cypress library
include(cypress_external)
include_directories(${CMAKE_SOURCE_DIR}/backend)

# BrainScaleS related stuff
#find_package(Boost REQUIRED serialization system filesystem mpi python27)

# Setup all includes, we use current github version for includes, which might break at some point -> BS do not deploy headers yet
execute_process(COMMAND "../setup_includes.sh"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set(BS_INCLUDE_PATH ${CMAKE_BINARY_DIR}/bs_includes) 
include_directories(
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${Boost_INCLUDE_DIRS} 
        ${PYTHON_INCLUDE_DIR}
        ${BS_INCLUDE_PATH}/euter/include
        ${BS_INCLUDE_PATH}/lib-boost-patches
        ${BS_INCLUDE_PATH}/marocco
        ${BS_INCLUDE_PATH}/pywrap/src
        ${BS_INCLUDE_PATH}/halbe
        ${BS_INCLUDE_PATH}/halco/include
        ${BS_INCLUDE_PATH}/ztl
        ${BS_INCLUDE_PATH}/rant
        ${BS_INCLUDE_PATH}/redman/include
        ${BS_INCLUDE_PATH}/sthal
        ${BS_INCLUDE_PATH}/pyublas/pyublas/include
        ${BS_INCLUDE_PATH}/calibtic/include
        ${BS_INCLUDE_PATH}/hicann-system/units/communication/source
        ${BS_INCLUDE_PATH}/hwdb
        ${BS_INCLUDE_PATH}/ester
)
# Set in BS Software, if not set, link will not work
add_compile_options(-D BOOST_VARIANT_DO_NOT_USE_VARIADIC_TEMPLATES)

# Manually set library path, because it is not included in LIBRARY_PATH environment variable
link_directories("/wang/environment/software/container/meta-nmpm-software/current/lib")
# Compile the actual executable
add_library(BS2CYPRESS SHARED
    backend/brainscales
)
add_dependencies(BS2CYPRESS cypress_ext)
target_link_libraries(BS2CYPRESS 
    PRIVATE ester-submit
    PRIVATE euter
    PRIVATE marocco
    PRIVATE marocco_runtime
    PRIVATE marocco_parameters
    PRIVATE marocco_coordinates
    PRIVATE marocco_results
    PRIVATE pymarocco_cpp
    PRIVATE halco_common
    PRIVATE halco_hicann_v2
    PRIVATE sthal
    PRIVATE halbe_container
    PRIVATE log4cxx.so.10
    PRIVATE ${Boost_LIBRARIES}
    PRIVATE ${PYTHON_LIBRARIES}
)


enable_testing()
add_subdirectory(test)
