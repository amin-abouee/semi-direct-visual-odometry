# relase binary folder
set(CMAKE_RELEASE_DIR "${CMAKE_SOURCE_DIR}/bin/release")
# debug binary folder
set(CMAKE_DEBUG_DIR "${CMAKE_SOURCE_DIR}/bin/debug")

find_package(OpenCV)
find_package(Eigen3)
find_package(Doxygen)
option(BUILD_DOC "Build documentation" OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)
# include_directories(${OpenCV_INCLUDE_DIRS})

set(LIBRARY_SOURCES
    feature-selection.cpp
    main.cpp
    )

# include_directories(math.hpp)
# add_library(source SHARED ${LIBRARY_SOURCES})

# set_target_properties(source PROPERTIES CMAKE_CXX_STANDARD_REQUIRED ON)
# set_target_properties(source PROPERTIES CXX_STANDARD 17)

add_executable(visual_odometry ${LIBRARY_SOURCES})
# target_link_libraries (visual_odometry visualizer source)
target_link_libraries (visual_odometry ${OpenCV_LIBS} Eigen3::Eigen)

set_target_properties(visual_odometry PROPERTIES CMAKE_CXX_STANDARD_REQUIRED ON)
set_target_properties(visual_odometry PROPERTIES CXX_STANDARD 17)


if(CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(visual_odometry PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_DEBUG_DIR})
	set_target_properties(visual_odometry PROPERTIES OUTPUT_NAME visual_odometry${BUILD_PREFIX})
else()
	set_target_properties(visual_odometry PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RELEASE_DIR})
endif()

if (BUILD_DOC)
   set(DOXYGEN_OUTPUT_FOLDER ${CMAKE_SOURCE_DIR}/docs/output)

    # set input and output files
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/docs/doxyfile.in)
    set(DOXYGEN_OUT ${DOXYGEN_OUTPUT_FOLDER}/doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    # message("Doxygen build started")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating HTML Documentation with Doxygen"
        VERBATIM )
endif (BUILD_DOC)