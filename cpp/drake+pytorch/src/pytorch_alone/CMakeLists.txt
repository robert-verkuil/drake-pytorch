cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch-alone)

#find_package(Torch REQUIRED)

add_executable(pytorch-alone pytorch-alone.cpp)
target_link_libraries(pytorch-alone drake::drake)
target_link_libraries(pytorch-alone "${TORCH_LIBRARIES}")
# target_link_libraries(pytorch-alone drake::drake "${TORCH_LIBRARIES}")
# set(LIBS drake::drake "${TORCH_LIBRARIES}")
# add_library(mytorch  "${TORCH_LIBRARIES}")
# set(LIBS drake::drake mytorch)
# target_link_libraries(pytorch-alone ${LIBS})
#set_property(TARGET pytorch-alone PROPERTY CXX_STANDARD 11)

add_test(NAME pytorch-alone
  COMMAND pytorch-alone
)
