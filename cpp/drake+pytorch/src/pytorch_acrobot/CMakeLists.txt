cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch-acrobot)


add_executable(pytorch-acrobot NNTestSetup.cpp NNSystem.cpp)
target_link_libraries(pytorch-acrobot drake::drake)
target_link_libraries(pytorch-acrobot "${TORCH_LIBRARIES}")

add_test(NAME pytorch-acrobot
  COMMAND pytorch-acrobot
)
