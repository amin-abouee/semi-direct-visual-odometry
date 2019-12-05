# https://github.com/onqtam/awesome-cmake
# https://cgold.readthedocs.io/en/latest/
cmake_minimum_required(VERSION 3.0.0)
project(semi-direct-visual-odometry VERSION 0.1.0)

# if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#     message(STATUS "GCC or Clang detected, adding compile flags")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# endif()

# message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
# message("CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
# message("CMAKE_CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
# message("CMAKE_CXX_FLAGS_MINSIZEREL: ${CMAKE_CXX_FLAGS_MINSIZEREL}")


# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

if(CMAKE_BUILD_TYPE STREQUAL Debug)
	set (BUILD_PREFIX "-d")
else()
	set (BUILD_PREFIX "")
endif()

add_subdirectory(src)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -pedantic -Werror -Wextra")
# string(REPLACE " " ";" REPLACED_FLAGS ${CMAKE_CXX_FLAGS})
# message(STATUS "REPLACED_FLAGS: ${REPLACED_FLAGS}")

# find_package(GTest REQUIRED)
# if(GTest_FOUND)
    # add_subdirectory(test)
# endif()
