# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/rverkuil/integration/integration/lib/python2.7/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/rverkuil/integration/integration/lib/python2.7/site-packages/cmake/data/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build

# Include any dependencies generated for this target.
include src/pytorch_alone/CMakeFiles/pytorch-alone.dir/depend.make

# Include the progress variables for this target.
include src/pytorch_alone/CMakeFiles/pytorch-alone.dir/progress.make

# Include the compile flags for this target's objects.
include src/pytorch_alone/CMakeFiles/pytorch-alone.dir/flags.make

src/pytorch_alone/CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.o: src/pytorch_alone/CMakeFiles/pytorch-alone.dir/flags.make
src/pytorch_alone/CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.o: ../src/pytorch_alone/pytorch-alone.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/pytorch_alone/CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.o"
	cd /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build/src/pytorch_alone && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.o -c /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/src/pytorch_alone/pytorch-alone.cpp

src/pytorch_alone/CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.i"
	cd /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build/src/pytorch_alone && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/src/pytorch_alone/pytorch-alone.cpp > CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.i

src/pytorch_alone/CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.s"
	cd /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build/src/pytorch_alone && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/src/pytorch_alone/pytorch-alone.cpp -o CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.s

# Object files for target pytorch-alone
pytorch__alone_OBJECTS = \
"CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.o"

# External object files for target pytorch-alone
pytorch__alone_EXTERNAL_OBJECTS =

src/pytorch_alone/pytorch-alone: src/pytorch_alone/CMakeFiles/pytorch-alone.dir/pytorch-alone.cpp.o
src/pytorch_alone/pytorch-alone: src/pytorch_alone/CMakeFiles/pytorch-alone.dir/build.make
src/pytorch_alone/pytorch-alone: /opt/drake/lib/libdrake.so
src/pytorch_alone/pytorch-alone: /home/rverkuil/integration/drake-pytorch/cpp/torchlib_cpu/lib/libtorch.so
src/pytorch_alone/pytorch-alone: /opt/drake/lib/libdrake_marker.so
src/pytorch_alone/pytorch-alone: /opt/drake/lib/libdrake_ignition_rndf0.so
src/pytorch_alone/pytorch-alone: /opt/drake/lib/libdrake_ignition_math4.so
src/pytorch_alone/pytorch-alone: /opt/drake/lib/libdrake_lcm.so
src/pytorch_alone/pytorch-alone: /home/rverkuil/integration/drake-pytorch/cpp/torchlib_cpu/lib/libprotobuf.a
src/pytorch_alone/pytorch-alone: /home/rverkuil/integration/drake-pytorch/cpp/torchlib_cpu/lib/libc10.so
src/pytorch_alone/pytorch-alone: src/pytorch_alone/CMakeFiles/pytorch-alone.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pytorch-alone"
	cd /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build/src/pytorch_alone && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pytorch-alone.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/pytorch_alone/CMakeFiles/pytorch-alone.dir/build: src/pytorch_alone/pytorch-alone

.PHONY : src/pytorch_alone/CMakeFiles/pytorch-alone.dir/build

src/pytorch_alone/CMakeFiles/pytorch-alone.dir/clean:
	cd /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build/src/pytorch_alone && $(CMAKE_COMMAND) -P CMakeFiles/pytorch-alone.dir/cmake_clean.cmake
.PHONY : src/pytorch_alone/CMakeFiles/pytorch-alone.dir/clean

src/pytorch_alone/CMakeFiles/pytorch-alone.dir/depend:
	cd /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/src/pytorch_alone /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build/src/pytorch_alone /home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/build/src/pytorch_alone/CMakeFiles/pytorch-alone.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/pytorch_alone/CMakeFiles/pytorch-alone.dir/depend

