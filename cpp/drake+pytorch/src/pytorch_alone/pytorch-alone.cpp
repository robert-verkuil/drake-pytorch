#include <torch/torch.h>
//#include <iostream>
//
//int main() {
//  torch::Tensor tensor = torch::rand({2, 3});
//  std::cout << tensor << std::endl;
//}

#include <iostream>
#include <stdexcept>

#include <drake/common/find_resource.h>

namespace shambhala {
namespace {

int main() {
  drake::FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/urdf/"
      "iiwa14_primitive_collision.urdf");

  try {
    drake::FindResourceOrThrow("nobody_home.urdf");
    std::cerr << "Should have thrown" << std::endl;
    return 1;
  } catch (const std::runtime_error&) {
  }

  return 0;
}

}  // namespace
}  // namespace shambhala

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  return shambhala::main();
}
