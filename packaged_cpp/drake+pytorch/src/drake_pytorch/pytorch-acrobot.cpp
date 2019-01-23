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
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/systems/primitives/linear_system.h"

#include "drake/solvers/snopt_solver.h"

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

namespace drake {
namespace systems {
namespace trajectory_optimization {
using trajectories::PiecewisePolynomial;
// namespace {

void do_traj_opt(){
  Eigen::Matrix2d A, B;
  A << 1, 2,
       3, 4;
  B << 5, 6,
       7, 8;
  // clang-format on
  const Eigen::MatrixXd C(0, 2), D(0, 2);
  const std::unique_ptr<LinearSystem<double>> system =
      std::make_unique<LinearSystem<double>>(A, B, C, D);
  const std::unique_ptr<Context<double>> context =
      system->CreateDefaultContext();

  const int kNumSampleTimes = 4;
  const double kTimeStep = .1;
  DirectCollocation prog(system.get(), *context, kNumSampleTimes, kTimeStep,
                         kTimeStep);

  prog.AddRunningCost(
      prog.state().cast<symbolic::Expression>().dot(prog.state()) +
      prog.input().cast<symbolic::Expression>().dot(prog.input()));

  Eigen::Matrix<double, 2, kNumSampleTimes> u;
  Eigen::Matrix<double, 2, kNumSampleTimes> x;
  for (int i = 0; i < kNumSampleTimes - 1; ++i) {
    prog.SetInitialGuess(prog.timestep(i), Vector1d(kTimeStep));
  }
  for (int i = 0; i < kNumSampleTimes; ++i) {
    x.col(i) << 0.2 * i - 1, 0.1 + i;
    u.col(i) << 0.1 * i, 0.2 * i + 0.1;
    prog.SetInitialGuess(prog.state(i), x.col(i));
    prog.SetInitialGuess(prog.input(i), u.col(i));
  }
  double total_cost = 0;
  for (const auto& cost : prog.GetAllCosts()) {
    total_cost += prog.EvalBindingAtInitialGuess(cost)(0);
  }
  const Eigen::Matrix<double, 1, kNumSampleTimes> g_val =
      (x.array() * x.array()).matrix().colwise().sum() +
      (u.array() * u.array()).matrix().colwise().sum();
  const double total_cost_expected =
      ((g_val.head<kNumSampleTimes - 1>() + g_val.tail<kNumSampleTimes - 1>()) /
       2 * kTimeStep)
          .sum();
  std::cout << "total_cost: " << total_cost << " total_cost_expected: " << total_cost_expected << std::endl;

  drake::solvers::SnoptSolver solver;
  std::cout << "is_thread_safe()? = " << solver.is_thread_safe() << std::endl;
  std::cout << "is_available()? = "   << solver.is_available() << std::endl;
  if (solver.available() && !solver.is_bounded_lp_broken()) {
    const auto solver_result = solver.Solve(prog);
  }

  // EXPECT_NEAR(total_cost, total_cost_expected, 1E-12);
}

// }  // anonymous namespace
}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace drake



int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  drake::systems::trajectory_optimization::do_traj_opt();

  return shambhala::main();
}




