cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch-acrobot)

# Acrobot setup
add_executable(pytorch-acrobot pytorch-acrobot.cpp)
target_link_libraries(pytorch-acrobot drake::drake)
target_link_libraries(pytorch-acrobot "${TORCH_LIBRARIES}")
add_test(NAME pytorch-acrobot
  COMMAND pytorch-acrobot
)

# # Acrobot setup
# add_executable(pytorch-acrobot NNSystem.cpp NNTestSetup.cpp)
# target_link_libraries(pytorch-acrobot drake::drake)
# target_link_libraries(pytorch-acrobot "${TORCH_LIBRARIES}")
# add_test(NAME pytorch-acrobot
#   COMMAND pytorch-acrobot
# )

# # Gradient Testing
# add_executable(pytorch-grad-test NNSystem.cpp NNGradientTest.cpp)
# target_link_libraries(pytorch-grad-test drake::drake)
# target_link_libraries(pytorch-grad-test "${TORCH_LIBRARIES}")
# add_test(NAME pytorch-grad-test
#   COMMAND pytorch-grad-test
# )

