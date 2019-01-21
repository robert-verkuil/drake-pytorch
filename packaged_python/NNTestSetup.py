import numpy as np
import time

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    ModelInstanceIndex,
    MultibodyPlant,
    PackageMap,
    Parser,
    SceneGraph,
    Simulator,
)

from networks import FCBIG
from NNSystem import NNSystem

def RenderSystemWithGraphviz(system, output_file="system_view.gz"):
    ''' Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. '''
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)


class NNTestSetup:
    def __init__(self, pytorch_nn_object=None):
        self.pytorch_nn_object = pytorch_nn_object

    def RunSimulation(self, real_time_rate=1.0):
        '''
        Here we test using the NNSystem in a Simulator to drive
        an acrobot.
        '''
        sdf_file = "assets/acrobot.sdf"
        urdf_file = "assets/acrobot.urdf"

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder)
        parser = Parser(plant=plant, scene_graph=scene_graph)
        parser.AddModelFromFile(sdf_file)
        plant.Finalize(scene_graph)

        # Add
        nn_system = NNSystem(self.pytorch_nn_object)
        builder.AddSystem(nn_system)

        # NN -> plant
        builder.Connect(nn_system.NN_out_output_port,
                        plant.get_actuation_input_port())
        # plant -> NN
        builder.Connect(plant.get_continuous_state_output_port(),
                        nn_system.NN_in_input_port)

        # Add meshcat visualizer
        meshcat = MeshcatVisualizer(scene_graph)
        builder.AddSystem(meshcat)
        # builder.Connect(scene_graph.GetOutputPort("lcm_visualization"),
        builder.Connect(scene_graph.get_pose_bundle_output_port(),
                        meshcat.GetInputPort("lcm_visualization"))

        # build diagram
        diagram = builder.Build()
        meshcat.load()
        # time.sleep(2.0)
        RenderSystemWithGraphviz(diagram)

        # construct simulator
        simulator = Simulator(diagram)

        # context = diagram.GetMutableSubsystemContext(
        #     self.station, simulator.get_mutable_context())


        simulator.set_publish_every_time_step(False)
        simulator.set_target_realtime_rate(real_time_rate)
        simulator.Initialize()
        sim_duration = 5.
        simulator.StepTo(sim_duration)
        print("stepping complete")


if __name__ == "__main__":
    net = FCBIG()
    nn_test_setup = NNTestSetup(pytorch_nn_object=net)
    nn_test_setup.RunSimulation()

