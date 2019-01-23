#include "drake/common/eigen_types.h"

#include "drake/common/find_resource.h"

#include "drake/geometry/dev/geometry_visualization.h"
#include "drake/geometry/dev/scene_graph.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"

#include "drake/lcm/drake_lcm.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"

#include "NNSystem.h"

#include <fstream>
#include <string>
#include <iostream>

// TODO: get this working
#include <torch/torch.h>


// TODO: convert to cpp?
void RenderSystemWithGraphviz(const drake::systems::System<double>& system, std::string output_file="system_view.gz"){
  /*
  * Renders the Drake system (presumably a diagram,
  * otherwise this graph will be fairly trivial) using
  * graphviz to a specified file. '''
  */
  std::string string = system.GetGraphvizString();
  std::ofstream out(output_file);
  out << string;
  out.close();
  std::cout << "graphviz string written to " << output_file << std::endl;
}

struct NonTorchFC: drake::systems::DrakeNet {
    // TODO: Implement this?
};

struct FC: drake::systems::DrakeNet {
  FC() {
    fc1 = register_module("fc1", torch::nn::Linear(4, 1));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = fc1->forward(x);
    return x;
  }

  torch::nn::Linear fc1{nullptr};
};

struct MLP: drake::systems::DrakeNet {
  MLP() {
    fc1 = register_module("fc1", torch::nn::Linear(4, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 1));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/true);
    x = torch::sigmoid(fc2->forward(x));
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};


namespace drake {
using drake::multibody::Parser;
using geometry::SceneGraph;
using geometry::ConnectDrakeVisualizer;
using multibody::MultibodyPlant;
namespace systems {


void TestInputGradients(bool autodiff_params=false){
    // Make net
    FC fc;
    MLP mlp;

    // Make system
    NNSystem<AutoDiffXd> nn_system{&mlp, /*declare_params*/autodiff_params};

    // Fix our derivatives coming into the system
    auto context = nn_system.CreateDefaultContext();
    VectorX<AutoDiffXd> autodiff_in{4};
    float values[] = {1., 3., 5., 7.};
    for (int i=0; i<4; i++){
        // Calculate gradients w.r.t. all gradients.
        autodiff_in[i] = AutoDiffXd(values[i], Vector1<double>::Constant(1.0));
    }
    context->FixInputPort(0, autodiff_in);

    // Optionally set AutoDiffXd's for parameters...
    if (autodiff_params){
        auto& params = context->get_mutable_numeric_parameter(0);
        for (int i=0; i<params.size(); i++){
            AutoDiffXd ad = params.GetAtIndex(i);
            params.SetAtIndex(i, AutoDiffXd(ad.value(), Vector1<double>::Constant(1.0)));
        }
    }

    std::unique_ptr<systems::SystemOutput<AutoDiffXd>> output =
        nn_system.AllocateOutput();
  
    // Check that the commanded pose starts out at zero, and that we can
    // set a different initial position.
    // Eigen::VectorXd expected = Eigen::VectorXd::Zero(kNumJoints * 2);
    nn_system.CalcOutput(*context, output.get());
    std::cout << "got: " << output->get_vector_data(0)->get_value() << std::endl;

      // Checks raw vector output.
//    EXPECT_TRUE(drake::CompareMatrices(expected_torque, dut_output, 1e-12,
//                                       drake::MatrixCompareType::absolute));
}

} // namespace systems
} // namespace drake

int main(){
    drake::systems::TestInputGradients(/*autodiff_params*/false);
    drake::systems::TestInputGradients(/*autodiff_params*/true);

    return 0;
}

