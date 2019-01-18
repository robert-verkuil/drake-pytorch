#include <torch/torch.h>
#include <iostream>

int main() {
    // Make our input, our weight matrix, output, and an empty jacobian matrix.
    auto x = torch::rand({2,1}, torch::TensorOptions().requires_grad(true));
    float M_data[] = {1,2,3,4};
    auto M = torch::from_blob(M_data, {2, 2});
    std::cout << M << std::endl;
    auto y = torch::mm(M, x);
    auto jacobian = torch::zeros({2, 2});

    // Backpropagate
    float tmp1[] = {1,0}, tmp2[] = {0,1};
    y.backward(torch::from_blob(tmp1, {2,1}), /*keep_graph*/true);
    std::cout << x.grad()<< std::endl; // jacobian[:,0] = x.grad.data

    // Clear out gradients from the last calculation
    x.grad().zero_();

    // Backpropagate again
    y.backward(torch::from_blob(tmp2, {2,1}), /*keep_graph*/true);
    std::cout << x.grad()<< std::endl; // jacobian[:,1] = x.grad.data

    return 0;
}

// #include <iostream>
// #include <stdexcept>
// 
// #include <drake/common/find_resource.h>
// 
// namespace shambhala {
// namespace {
// 
// int main() {
//   drake::FindResourceOrThrow(
//       "drake/manipulation/models/iiwa_description/urdf/"
//       "iiwa14_primitive_collision.urdf");
// 
//   try {
//     drake::FindResourceOrThrow("nobody_home.urdf");
//     std::cerr << "Should have thrown" << std::endl;
//     return 1;
//   } catch (const std::runtime_error&) {
//   }
// 
//   return 0;
// }
// 
// }  // namespace
// }  // namespace shambhala
// 
// int main() {
//   torch::Tensor tensor = torch::rand({2, 3});
//   std::cout << tensor << std::endl;
//   return shambhala::main();
// }
