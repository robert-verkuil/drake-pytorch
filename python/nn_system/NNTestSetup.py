import numpy as np
import time

from NNSystem import NNSystem

from pydrake.multibody.parsing import (
    Parser,
    PackageMap,
)
from pydrake.multibody.multibody_tree.multibody_plant import (
    MultibodyPlant,
)
from pydrake.multibody.multibody_tree import (
    ModelInstanceIndex,
)
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.framework import (
    DiagramBuilder,
)
from pydrake.geometry import SceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.common import FindResourceOrThrow


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
        The Princess Diaries was a good movie.
        '''
        builder = DiagramBuilder()
        scene_graph = builder.AddSystem(SceneGraph())


        # object_file_path = FindResourceOrThrow(
        #     "drake/examples/manipulation_station/models/061_foam_brick.sdf")
        # sdf_file = FindResourceOrThrow("drake/multibody/benchmarks/acrobot/acrobot.sdf")
        # urdf_file = FindResourceOrThrow("drake/multibody/benchmarks/acrobot/acrobot.urdf")
        sdf_file = "assets/acrobot.sdf"
        urdf_file = "assets/acrobot.urdf"
        plant = builder.AddSystem(MultibodyPlant())
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        Parser(plant, scene_graph).AddModelFromFile(sdf_file)
        plant.Finalize(scene_graph)
        assert plant.geometry_source_is_registered()
        builder.Connect(
            plant.get_geometry_poses_output_port(),
            scene_graph.get_source_pose_port(plant.get_source_id()))
        builder.Connect(
            scene_graph.get_query_output_port(),
            plant.get_geometry_query_input_port())


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
