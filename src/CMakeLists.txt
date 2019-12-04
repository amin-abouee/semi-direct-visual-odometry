# relase binary folder
set(CMAKE_RELEASE_DIR "${CMAKE_SOURCE_DIR}/bin/release")
# debug binary folder
set(CMAKE_DEBUG_DIR "${CMAKE_SOURCE_DIR}/bin/debug")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(nlohmann_json REQUIRED)

# https://github.com/microsoft/LightGBM/issues/1898
# cmake \
#   -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
#   -DOpenMP_C_LIB_NAMES="omp" \
#   -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
#   -DOpenMP_CXX_LIB_NAMES="omp" \
#   -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib \
#   ..

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set (OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set (OpenMP_C_LIB_NAMES "omp")
    set (OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set (OpenMP_CXX_LIB_NAMES "omp")
    set (OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
endif()

# https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
# https://stackoverflow.com/questions/56202041/compiling-and-linking-against-openmp-with-appleclang-on-mac-os-x-mojave
find_package(OpenMP REQUIRED)

# https://github.com/gabime/spdlog/blob/v1.x/example/CMakeLists.txt
find_package(spdlog REQUIRED)
find_package(Threads)
find_package(Doxygen)
option(BUILD_DOC "Build documentation" OFF)

# https://github.com/xianyi/OpenBLAS/wiki/Faq#debianlts
# https://github.com/xianyi/OpenBLAS/issues/1394
# https://stackoverflow.com/questions/38350862/cmake-findlapack-cmake-and-findblas-cmake-can-not-link-lapack-and-blas
option(USE_BLAS_EIGEN "Use OpenBLAS and OpenLapack as backend for Eigen" OFF)
if(USE_BLAS_EIGEN)
    find_package(BLAS REQUIRED)
    # find_package(LAPACK REQUIRED)
    # add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
    add_definitions(-DEIGEN_USE_BLAS)
endif()


include_directories(${CMAKE_SOURCE_DIR}/include
                    ${Sophus_INCLUDE_DIR})
# include_directories(${OpenCV_INCLUDE_DIRS})

set(LIBRARY_SOURCES
    pinhole_camera.cpp
    frame.cpp
    image_pyramid.cpp
    feature.cpp
    matcher.cpp
    feature_selection.cpp
    algorithm.cpp
    visualization.cpp
    main.cpp
    )


# include_directories(math.hpp)
# add_library(source SHARED ${LIBRARY_SOURCES})

# set_target_properties(source PROPERTIES CMAKE_CXX_STANDARD_REQUIRED ON)
# set_target_properties(source PROPERTIES CXX_STANDARD 17)

add_executable(visual_odometry ${LIBRARY_SOURCES})
# target_link_libraries (visual_odometry visualizer source)
target_link_libraries (visual_odometry PRIVATE ${OpenCV_LIBS} Eigen3::Eigen nlohmann_json::nlohmann_json Threads::Threads spdlog::spdlog)
if(OpenMP_CXX_FOUND)
    target_link_libraries(visual_odometry PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_BLAS_EIGEN AND BLAS_FOUND )
    message( ${BLAS_LIBRARIES})
    # target_link_libraries(visual_odometry PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_link_libraries(visual_odometry PRIVATE  ${BLAS_LIBRARIES})
endif()


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