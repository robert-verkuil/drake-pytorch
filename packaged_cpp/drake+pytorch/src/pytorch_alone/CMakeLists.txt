cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch-alone)

add_executable(pytorch-alone pytorch-alone.cpp)
target_link_libraries(pytorch-alone drake::drake)
target_link_libraries(pytorch-alone "${TORCH_LIBRARIES}")

add_test(NAME pytorch-alone
  COMMAND pytorch-alone
)
