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
// #include <torch/torch.h>


namespace drake {
using drake::multibody::Parser;
using geometry::SceneGraph;
using geometry::ConnectDrakeVisualizer;
using multibody::MultibodyPlant;


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


class NNTestSetup
{
  public:
    NNTestSetup(drake::systems::DrakeNet *neural_network)
      : neural_network_(neural_network) {
        std::cout << "babies first constructor " << "replaceThis" << std::endl;
    }

    void RunSimulation(float real_time_rate=1.0){
        /*
        * Desc. here
        */
        systems::DiagramBuilder<double> builder;
        SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();

        const char sdfPath[] = "/home/rverkuil/integration/drake-pytorch/cpp/drake+pytorch/src/pytorch_acrobot/assets/acrobot.sdf"; // TODO: handle needing full paths better than this!
        const char urdfPath[] = "assets/acrobot.urdf";

        MultibodyPlant<double>& plant = *builder.AddSystem<MultibodyPlant>();
        plant.RegisterAsSourceForSceneGraph(&scene_graph);
        Parser(&plant, &scene_graph).AddModelFromFile(
            sdfPath);

        // It's one of these next two lines.
        plant.Finalize(&scene_graph);
        // plant.Finalize()
        DRAKE_DEMAND(plant.geometry_source_is_registered());

        // These might just work?
        builder.Connect(
            plant.get_geometry_poses_output_port(),
            scene_graph.get_source_pose_port(plant.get_source_id().value()));
        builder.Connect(
            scene_graph.get_query_output_port(),
            plant.get_geometry_query_input_port());

        // Add
        //drake::systems::NNSystem nn_system{neural_network_};
        auto nn_system = builder.AddSystem<drake::systems::NNSystem>(neural_network_);

        // NN -> plant
        builder.Connect(nn_system->GetOutputPort("NN_out"),
                        plant.get_actuation_input_port());
        // plant -> NN
        builder.Connect(plant.get_continuous_state_output_port(),
                        nn_system->GetInputPort("NN_in"));

        // Add visualizer
        ConnectDrakeVisualizer(&builder, scene_graph);

        lcm::DrakeLcm lcm;
        lcm.StartReceiveThread();

        // build diagram
        auto diagram = builder.Build();
        // time.sleep(2.0);
        RenderSystemWithGraphviz(*diagram);

        // construct simulator
        systems::Simulator<double> simulator(*diagram);

        simulator.set_publish_every_time_step(false);
        simulator.set_target_realtime_rate(real_time_rate);
        simulator.Initialize();
      float sim_duration = 5.;
      simulator.StepTo(sim_duration);
    }
    private:
      drake::systems::DrakeNet *neural_network_;
};
} // namespace drake


// Define a new Module.
// struct Net : torch::nn::Module {
//struct Net : DrakeNet {
//  Net() {
//    // Construct and register two Linear submodules.
//    fc1 = register_module("fc1", torch::nn::Linear(8, 64));
//    fc2 = register_module("fc2", torch::nn::Linear(64, 1));
//  }
//
//  // Implement the Net's algorithm.
//  torch::Tensor forward(torch::Tensor x) {
//    // Use one of many tensor manipulation functions.
//    x = torch::relu(fc1->forward(x));
//    x = torch::dropout(x, /*p=*/0.5, /*train=*/true);
//    x = torch::sigmoid(fc2->forward(x));
//    return x;
//  }
//
//  // Use one of many "standard library" modules.
//  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//};

struct Net : drake::systems::DrakeNet {
  Net() {
    // Construct and register two Linear submodules.
      std::cout << "Net constructed." << std::endl;
  }

  // Implement the Net's algorithm.
  int forward() {
    return 1;
  }
};

int main(){
    // Create a new Net.
    Net net;
    auto nnTest = drake::NNTestSetup{&net};
    //drake::NNTestSetup{1776};
    nnTest.RunSimulation(); // <- Get this working!!!
    return 0;
}

