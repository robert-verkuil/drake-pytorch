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

class NNGradTest
{
  public:
    NNGradTest(DrakeNet *neural_network, int n_inputs, int n_outputs)
      : neural_network_(neural_network),
        n_inputs_(n_inputs),
        n_outputs_(n_outputs) {
    }

    void Test1(float real_time_rate=1.0){
//        systems::DiagramBuilder<double> builder;
//        SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
//
//        // Add
//        auto nn_system = builder.AddSystem<NNSystem>(neural_network_, n_inputs_, n_outputs_);
//
//        // build diagram
//        auto diagram = builder.Build();
//        RenderSystemWithGraphviz(*diagram);

        // The plant and its context.
        NNSystem<AutoDiffXd> system{neural_network_, n_inputs_, n_outputs_};
        auto context = system.CreateDefaultContext();
        context->FixInputPort(0, Vector1<AutoDiffXd>::Constant(0.0));

    }
    private:
      DrakeNet *neural_network_;
      int n_inputs_;
      int n_outputs_;
};
} // namespace systems
} // namespace drake

int main(){
    FC net;
    //MLP net;
    auto nnGradTest = drake::systems::NNGradTest{&net, 4, 1};
    nnGradTest.Test1();

    return 0;
}

