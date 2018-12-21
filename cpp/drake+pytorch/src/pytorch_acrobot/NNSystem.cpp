#include "NNSystem.h"

#include "drake/common/default_scalars.h"
#include "drake/systems/framework/basic_vector.h"

//#include <torch/torch.h>

namespace drake {
namespace systems {

NNSystem::NNSystem(DrakeNet *neural_network)
    //: LeafSystem<double>(SystemTypeTag<systems::NNSystem>{}), // Will need to enable this again!
    : //LeafSystem<double>(SystemTypeTag<NNSystem>{}),
      neural_network_(neural_network) {

  // Declare ports.
  this->DeclareInputPort("NN_in", kVectorValued, 4); // TODO figure out num_inputs automatically
  this->DeclareVectorOutputPort("NN_out", BasicVector<double>(1), // TODO figure out num_outputs automatically
                                &NNSystem::Forward);
}

// Copy constructor?
//template <typename T>
//template <typename U>
//NNSystem<T>::NNSystem(const NNSystem<U>& other)
//    : NNSystem<T>(other.get_num_input_ports(), other.get_input_port(0).size()) {}

// The NN inference method.
void NNSystem::Forward(const Context<double>& context,
                       BasicVector<double>* out) const {
  Eigen::VectorBlock<VectorX<double>> out_vector = out->get_mutable_value();

  // Zeroes the output.
  out_vector.setZero();

  // Sum each input port into the output.
  const BasicVector<double>* input_vector = this->EvalVectorInput(context, 0);
  // Have to convert input to a torch tensor
//  torch::Tensor in = torch::ones({8});
//  torch::Tensor torch_out = neural_network_->forward(in); // TODO: give an actual input here.
  // Have to put this into a basicvector somehow? TODO: Use Eigen here!!
  // *out = ...
}

}  // namespace systems
}  // namespace drake

// Should eventually use the non-symbolic version here!
//DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
//    class ::drake::systems::NNSystem)
