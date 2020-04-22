# http://www.brianlheim.com/2018/04/09/cmake-cheat-sheet.html
# http://manpages.ubuntu.com/manpages/xenial/man7/cmake-generator-expressions.7.html
# https://www.systutorials.com/docs/linux/man/1-cmakecommands/

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

# if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#     set (OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
#     set (OpenMP_C_LIB_NAMES "omp")
#     set (OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
#     set (OpenMP_CXX_LIB_NAMES "omp")
#     set (OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
# endif()

# https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
# https://stackoverflow.com/questions/56202041/compiling-and-linking-against-openmp-with-appleclang-on-mac-os-x-mojave
# find_package(OpenMP REQUIRED)

# https://github.com/gabime/spdlog/blob/v1.x/example/CMakeLists.txt
# find_package(spdlog)
# find_package(EASYLOGGINGPP REQUIRED)
find_package(Threads)

# https://github.com/lava/matplotlib-cpp
find_package(Python2 COMPONENTS Development NumPy)

find_package(Doxygen)
option(BUILD_DOC "Build documentation" OFF)

# https://github.com/xianyi/OpenBLAS/wiki/Faq#debianlts
# https://github.com/xianyi/OpenBLAS/issues/1394
# https://stackoverflow.com/questions/38350862/cmake-findlapack-cmake-and-findblas-cmake-can-not-link-lapack-and-blas
option(USE_BLAS_EIGEN "Use OpenBLAS as backend for Eigen" OFF)
if(USE_BLAS_EIGEN)
    find_package(BLAS REQUIRED)
    # find_package(LAPACK REQUIRED)
    # add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
    # add_definitions(-DEIGEN_USE_BLAS)
endif()

option(COMPILE_WITH_OFAST "Compile with -Ofast optimization" OFF)
# include_directories(${CMAKE_SOURCE_DIR}/include
#                     ${Sophus_INCLUDE_DIR})


# https://github.com/toeb/moderncmake/blob/master/Modern%20CMake.pdf
set (LIBRARY_INCLUDE
    ${CMAKE_SOURCE_DIR}/include
    ${Sophus_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/3rd_party/easy_logging
    )

set(LIBRARY_SOURCES
    pinhole_camera.cpp
    frame.cpp
    image_pyramid.cpp
    feature.cpp
    point.cpp
    map.cpp
    depth_filter.cpp
    image_alignment.cpp
    optimizer.cpp
    estimator.cpp
    matcher.cpp
    feature_selection.cpp
    algorithm.cpp
    visualization.cpp
    config.cpp
    system.cpp
    utils.cpp
    main.cpp
    )

# add_library(source SHARED ${LIBRARY_SOURCES})

# Specify compile definitions to use when compiling a given <target>. 
# The named <target> must have been created by a command such as add_executable() or add_library() and must not be an Imported Target.
add_executable(visual_odometry ${LIBRARY_SOURCES} ${CMAKE_SOURCE_DIR}/3rd_party/easy_logging/easylogging++.cc)

# https://clang.llvm.org/docs/CommandGuide/clang.html#code-generation-options
# https://stackoverflow.com/questions/15548023/clang-optimization-levels/15548189#15548189
# https://stackoverflow.com/a/22135559/1804533
# -DNDEBUG is the macro to be defined to turn off asserts as mandated by the C standard.
# set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -march=native")
if(COMPILE_WITH_OFAST)
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG ")
    # https://stackoverflow.com/a/23995391/1804533
    # target_compile_options(visual_odometry PRIVATE "$<$<CONFIG:RELEASE>:${CMAKE_CXX_FLAGS_RELEASE}>")
endif()

target_compile_options(visual_odometry PRIVATE "$<$<CONFIG:RELEASE>:-march=native>")
# target_compile_options(visual_odometry PRIVATE "$<$<CONFIG:RELEASE>:-stdlib=libc++ -lc++abi>")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

# https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
target_compile_definitions(visual_odometry PRIVATE -DEIGEN_MAX_CPP_VER=17 )

# https://stackoverflow.com/a/31969632/1804533
# target_include_directories(visual_odometry PRIVATE ${LIBRARY_INCLUDE})

# add matplotlibcpp
target_include_directories(visual_odometry PRIVATE ${LIBRARY_INCLUDE} ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})

# target_link_libraries (visual_odometry visualizer source)
# spdlog::spdlog
target_link_libraries (visual_odometry PRIVATE ${OpenCV_LIBS} Eigen3::Eigen nlohmann_json::nlohmann_json Threads::Threads)

# http://www.brianlheim.com/2018/04/09/cmake-cheat-sheet.html
# http://manpages.ubuntu.com/manpages/xenial/man7/cmake-generator-expressions.7.html
# $<$<VERSION_LESS:$<CXX_COMPILER_VERSION>,4.2.0>:OLD_COMPILER>
target_link_libraries(visual_odometry PRIVATE $<$<CXX_COMPILER_ID:GNU>:stdc++fs>)
# target_link_libraries(visual_odometry PRIVATE stdc++fs)

# target_link_libraries (visual_odometry PRIVATE stdc++fs)
target_link_libraries (visual_odometry PRIVATE Python2::Python Python2::NumPy)

if(OpenMP_CXX_FOUND)
    # target_link_libraries(visual_odometry PRIVATE OpenMP::OpenMP_CXX)
endif()

# https://stackoverflow.com/a/27558697/1804533
if(USE_BLAS_EIGEN AND BLAS_FOUND )
    message( STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
    # message( STATUS "LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")
    # https://stackoverflow.com/a/15263203/1804533
    target_compile_definitions(visual_odometry PRIVATE -DEIGEN_USE_BLAS )
    # target_link_libraries(visual_odometry PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_link_libraries(visual_odometry PRIVATE  ${BLAS_LIBRARIES})
endif()

# https://github.com/ricab/scope_guard/issues/4#issuecomment-387529287
# https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
# https://clang.llvm.org/docs/DiagnosticsReference.html
# https://foonathan.net/2018/10/cmake-warnings/
# http://www.brianlheim.com/2018/04/09/cmake-cheat-sheet.html

# -Wall -Wextra -pedantic -Wconversion -Wsign-conversion

target_compile_options(visual_odometry
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Wextra -pedantic
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        W4
    >
)
# set_target_properties(visual_odometry PROPERTIES CMAKE_CXX_STANDARD_REQUIRED ON)
# set_target_properties(visual_odometry PROPERTIES CMAKE_CXX_EXTENSIONS ON)
# set_target_properties(visual_odometry PROPERTIES CXX_STANDARD 17)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")


#*******
# https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html
# https://cmake.org/cmake/help/latest/prop_gbl/CMAKE_CXX_KNOWN_FEATURES.html#prop_gbl:CMAKE_CXX_KNOWN_FEATURES
#*******

# https://crascit.com/2015/03/28/enabling-cxx11-in-cmake/
target_compile_features(visual_odometry PUBLIC cxx_std_17)


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