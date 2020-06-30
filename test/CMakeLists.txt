#  Cypress -- C++ Spiking Neural Network Simulation Framework
#  Copyright (C) 2019  Christoph Ostrau
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
cmake_minimum_required (VERSION 3.0)
project (BS2CYPRESS)

include_directories(
	PRIVATE
		${GTEST_INCLUDE_DIRS}
)

add_executable(test_brainscales
	test_brainscales.cpp
)
add_dependencies(test_brainscales cypress_ext BS2CYPRESS)
target_link_libraries(test_brainscales PUBLIC
    BS2CYPRESS
	${CMAKE_BINARY_DIR}/cypress_ext-prefix/src/cypress_ext-build/libcypress.a
    PRIVATE euter
    PRIVATE marocco
    PRIVATE marocco_runtime
    PRIVATE marocco_parameters
    PRIVATE marocco_coordinates
    PRIVATE marocco_results
    PRIVATE pymarocco_cpp
    PRIVATE halco_hicann_v2
    PRIVATE halco_common
    PRIVATE halco_hicann_v2
    PRIVATE sthal
    PRIVATE halbe_container
    PRIVATE log4cxx.so.10
     #PRIVATE ${Boost_LIBRARIES}
     PRIVATE ${PYTHON_LIBRARIES}
    dl
	${GTEST_LIBRARIES}
	pthread
)
add_test(test_brainscales test_brainscales)
