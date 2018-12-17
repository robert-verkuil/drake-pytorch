# - Config file for the Caffe2 package
# It defines the following variable(s)
#   CAFFE2_INCLUDE_DIRS     - include directories for FooBar
# as well as Caffe2 targets for other cmake libraries to use.

# library version information

set(CAFFE2_VERSION_MAJOR 1)
set(CAFFE2_VERSION_MINOR 0)
set(CAFFE2_VERSION_PATCH 0)
set(CAFFE2_VERSION "1.0.0")

# Utils functions.
include("${CMAKE_CURRENT_LIST_DIR}/public/utils.cmake")

# Include threads lib.
include("${CMAKE_CURRENT_LIST_DIR}/public/threads.cmake")

# Depending on whether Caffe2 uses gflags during compile time or
# not, invoke gflags.
if (OFF)
  include("${CMAKE_CURRENT_LIST_DIR}/public/gflags.cmake")
  if (NOT TARGET gflags)
    message(FATAL_ERROR
        "Your installed Caffe2 version uses gflags but the gflags library "
        "cannot be found. Did you accidentally remove it, or have you set "
        "the right CMAKE_PREFIX_PATH and/or GFLAGS_ROOT_DIR? If you do not "
        "have gflags, you will need to install gflags and set the library "
        "path accordingly.")
  endif()
endif()

# Depending on whether Caffe2 uses glog during compile time or
# not, invoke glog.
if (OFF)
  include("${CMAKE_CURRENT_LIST_DIR}/public/glog.cmake")
  if (NOT TARGET glog::glog)
    message(FATAL_ERROR
        "Your installed Caffe2 version uses glog but the glog library "
        "cannot be found. Did you accidentally remove it, or have you set "
        "the right CMAKE_PREFIX_PATH and/or GFLAGS_ROOT_DIR? If you do not "
        "have glog, you will need to install glog and set the library "
        "path accordingly.")
  endif()
endif()

# Protobuf
if (ON)
  if (NOT TARGET protobuf::libprotobuf)
    # Define protobuf::libprotobuf as a dummy target to resolve references to
    # protobuf::libprotobuf in Caffe2Targets.cmake.
    add_library(dummy INTERFACE)
    add_library(protobuf::libprotobuf ALIAS dummy)
  endif()
else()
  include("${CMAKE_CURRENT_LIST_DIR}/public/protobuf.cmake")
  if (NOT TARGET protobuf::libprotobuf)
    message(FATAL_ERROR
        "Your installed Caffe2 version uses protobuf but the protobuf library "
        "cannot be found. Did you accidentally remove it, or have you set "
        "the right CMAKE_PREFIX_PATH? If you do not have protobuf, you will "
        "need to install protobuf and set the library path accordingly.")
  endif()
  message(STATUS "Caffe2: Protobuf version " ${Protobuf_VERSION})
  # If during build time we know the protobuf version, we will also do a sanity
  # check to ensure that the protobuf library that Caffe2 found is consistent
  # with the compiled version.
  if (FALSE)
    if (NOT (${Protobuf_VERSION} VERSION_EQUAL Protobuf_VERSION_NOTFOUND))
      message(FATAL_ERROR
          "Your installed Caffe2 is built with protobuf "
          "Protobuf_VERSION_NOTFOUND"
          ", while your current cmake setting discovers protobuf version "
          ${Protobuf_VERSION}
          ". Please specify a protobuf version that is the same as the built "
          "version.")
    endif()
  endif()
endif()

if (0)
  # The file public/cuda.cmake exclusively uses CAFFE2_USE_*.
  # If Caffe2 was compiled with the libraries below, they must
  # be found again when including the Caffe2 target.
  set(CAFFE2_USE_CUDA 0)
  set(CAFFE2_USE_CUDNN OFF)
  set(CAFFE2_USE_TENSORRT OFF)
  include("${CMAKE_CURRENT_LIST_DIR}/public/cuda.cmake")
  if ( AND NOT CAFFE2_USE_CUDA)
    message(FATAL_ERROR
      "Your installed Caffe2 version uses CUDA but I cannot find the CUDA "
      "libraries. Please set the proper CUDA prefixes and / or install "
      "CUDA.")
  endif()
  if ( AND NOT CAFFE2_USE_CUDNN)
    message(FATAL_ERROR
      "Your installed Caffe2 version uses cuDNN but I cannot find the cuDNN "
      "libraries. Please set the proper cuDNN prefixes and / or install "
      "cuDNN.")
  endif()
  if ( AND NOT CAFFE2_USE_TENSORRT)
    message(FATAL_ERROR
      "Your installed Caffe2 version uses TensorRT but I cannot find the TensorRT "
      "libraries. Please set the proper TensorRT prefixes and / or install "
      "TensorRT.")
  endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/public/mkl.cmake")

if (1)
  include("${CMAKE_CURRENT_LIST_DIR}/public/mkldnn.cmake")
endif()

# import targets
include ("${CMAKE_CURRENT_LIST_DIR}/Caffe2Targets.cmake")

# Interface libraries, that allows one to build proper link flags.
# We will also define a helper variable, Caffe2_MAIN_LIBS, that resolves to
# the main caffe2 libraries in cases of cuda presence / absence.
caffe2_interface_library(caffe2 caffe2_library)
if (0)
  caffe2_interface_library(caffe2_gpu caffe2_gpu_library)
  set(Caffe2_MAIN_LIBS caffe2_library caffe2_gpu_library)
else()
  set(Caffe2_MAIN_LIBS caffe2_library)
endif()

# include directory.
#
# Newer versions of CMake set the INTERFACE_INCLUDE_DIRECTORIES property
# of the imported targets. It is hence not necessary to add this path
# manually to the include search path for targets which link to gflags.
# The following lines are here for backward compatibility, in case one
# would like to use the old-style include path.
get_filename_component(
    CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
# Note: the current list dir is _INSTALL_PREFIX/share/cmake/Gloo.
get_filename_component(
    _INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
set(CAFFE2_INCLUDE_DIRS "${_INSTALL_PREFIX}/include")
