#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/expect_throws_message.h" // TODO: needed?

#include "drake/geometry/dev/geometry_visualization.h"
#include "drake/geometry/dev/scene_graph.h"

#include "drake/lcm/drake_lcm.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"

// TODO: Make sure to put these inside of the namespacing brackets...
using geometry::dev::ConnectDrakeVisualizer;
using geometry::dev::SceneGraph;
using geometry::SourceId;
using lcm::DrakeLcm;

using drake::multibody::MultibodyPlant;
using geometry::SceneGraph;

#include "NNSystem.h"

// TODO: get this working
// #include <torch/torch.h>


// TODO: convert to cpp?
// def RenderSystemWithGraphviz(system, output_file="system_view.gz"):
//     /*
//     * Renders the Drake system (presumably a diagram,
//     * otherwise this graph will be fairly trivial) using
//     * graphviz to a specified file. '''
//     */
//     from graphviz import Source
//     string = system.GetGraphvizString()
//     src = Source(string)
//     src.render(output_file, view=False)

// Random Rob notes:
// Seems like systems are uppercase concated, and templated by double?

class NNTestSetup
{
    // def __init__(self, pytorch_nn_object=None):
    //     self.pytorch_nn_object = pytorch_nn_object
    NNTestSetup::NNTestSetup(int replaceThis){
        std::cout << "babies first constructor " << replaceThis << std::endl;
    }

    NNTestSetup::RunSimulation(self, real_time_rate=1.0){
        /*
        * Desc. here
        */
        systems::DiagramBuilder<double> builder;
        SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();

        const char sdfPath[] = "assets/acrobot.sdf";
        const char urdfPath[] = "assets/acrobot.urdf":

        MultibodyPlant<double>& plant = *builder.AddSystem<MultibodyPlant>();
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        Parser(&plant, &scene_graph).AddModelFromFile(
            FindResourceOrThrow(sdfPath));

        // It's one of these next two lines.
        plant.Finalize(scene_graph)
        // plant.Finalize()
        DRAKE_DEMAND(plant.geometry_source_is_registered())

        // These might just work?
        builder.Connect(
            plant.get_geometry_poses_output_port(),
            scene_graph.get_source_pose_port(plant.get_source_id()))
        builder.Connect(
            scene_graph.get_query_output_port(),
            plant.get_geometry_query_input_port())

        // BELOW HERE NO IDEA IF IT WORKS...
        
        // Add
        nn_system = NNSystem(self.pytorch_nn_object)
        builder.AddSystem(nn_system)

        // NN -> plant
        builder.Connect(nn_system.NN_out_output_port,
                        plant.get_actuation_input_port())
        // plant -> NN
        builder.Connect(plant.get_continuous_state_output_port(),
                        nn_system.NN_in_input_port)


        // Add meshcat visualizer
        // TODO: use ConnectDrakeVisualizer(&builder, *scene_graph);
        meshcat = MeshcatVisualizer(scene_graph)
        builder.AddSystem(meshcat)
        builder.Connect(scene_graph.get_pose_bundle_output_port(),
                        meshcat.GetInputPort("lcm_visualization"))

        // build diagram
        diagram = builder.Build()
        meshcat.load()
        // time.sleep(2.0)
        RenderSystemWithGraphviz(diagram)

        // construct simulator
        simulator = Simulator(diagram)

        simulator.set_publish_every_time_step(False)
        simulator.set_target_realtime_rate(real_time_rate)
        simulator.Initialize()
        sim_duration = 5.
        simulator.StepTo(sim_duration)
    }
}


// // Define a new Module.
// struct Net : torch::nn::Module {
//   Net() {
//     // Construct and register two Linear submodules.
//     fc1 = register_module("fc1", torch::nn::Linear(8, 64));
//     fc2 = register_module("fc2", torch::nn::Linear(64, 1));
//   }
// 
//   // Implement the Net's algorithm.
//   torch::Tensor forward(torch::Tensor x) {
//     // Use one of many tensor manipulation functions.
//     x = torch::relu(fc1->forward(x));
//     x = torch::dropout(x, /*p=*/0.5, /*train=*/true);
//     x = torch::sigmoid(fc2->forward(x));
//     return x;
//   }
// 
//   // Use one of many "standard library" modules.
//   torch::nn::Linear fc1{nullptr}, fc2{nullptr};
// };


void main(){
    // Create a new Net.
//    Net net;
    const int net = 5; // TODO: remove this
    auto nnTest = NNTestSetup(net);
    // nnTest.runSimulation() # <- Get this working!!!
    return 0;
}
