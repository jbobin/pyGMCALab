cmake_minimum_required(VERSION 2.8)

include(ExternalProject)
include(FindPkgConfig)

# Adding customized cmake module
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(pyredwave)

# Find required packages
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonInterp REQUIRED)
find_package(PythonLibsNew REQUIRED)

#set(BOOST_ROOT /Users/jbobin/Downloads/boost_1_57_0)
set(Boost_INCLUDE_DIR ${BOOST_ROOT}/boost)
set(Boost_LIBRARY_DIR ${BOOST_ROOT}/stage/lib)

# Downloads and compiles the Boost.NumPy package
# The library and include files are located in the build/extern directory
ExternalProject_Add(BoostNumpy
    GIT_REPOSITORY https://github.com/ndarray/Boost.NumPy
    CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/extern
                   -DLIBRARY_TYPE=STATIC
)
set(BoostNumpy_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/extern/include)
set(BoostNumpy_LIBRARY_DIRS ${CMAKE_BINARY_DIR}/extern/lib)
set(BoostNumpy_LIBRARIES -lboost_numpy)


# Define include and library directories
include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${BoostNumpy_INCLUDE_DIRS}
)
link_directories(${BoostNumpy_LIBRARY_DIRS})


# Compilation flags
set(CMAKE_CXX_FLAGS "-O3 -fomit-frame-pointer -fno-common -fPIC -fopenmp -DRW_PYTHON_INTERFACE -F/Library/Frameworks -arch x86_64")


add_library(redwavecxx SHARED pyredwave/cxx/redWaveTools.cpp pyredwave/cxx/redwavecxx.cpp)
add_dependencies(redwavecxx BoostNumpy)
target_link_libraries(redwavecxx ${BoostNumpy_LIBRARIES} ${Boost_LIBRARIES} ${GSL_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(redwavecxx PROPERTIES PREFIX "")
set_target_properties(redwavecxx PROPERTIES SUFFIX .so)
if(CYGWIN OR MSYS OR MINGW)
	# Ensure the output binary is called redwavecxx.pyd on Windows (not redwavecxx.dll)
	set_target_properties(redwavecxx PROPERTIES SUFFIX ".pyd")
endif(CYGWIN OR MSYS OR MINGW)

# Installs the module inside the python package 
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

install(TARGETS redwavecxx DESTINATION pyredwave)

