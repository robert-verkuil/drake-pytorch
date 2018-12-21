#include "NNSystem.h"

#include "drake/common/default_scalars.h"
#include "drake/systems/framework/basic_vector.h"

//#include <torch/torch.h>

namespace drake {
namespace systems {

template <typename T>
NNSystem<T>::NNSystem(DrakeNet *neural_network)
    : LeafSystem<T>(SystemTypeTag<systems::NNSystem>{}),
      neural_network_(neural_network) {

  // Declare ports.
  this->DeclareInputPort("input", kVectorValued, 4); // TODO figure out num_inputs automatically
  this->DeclareVectorOutputPort("output", BasicVector<T>(1), // TODO figure out num_outputs automatically
                                &NNSystem<T>::Forward);
}

// Copy constructor?
//template <typename T>
//template <typename U>
//NNSystem<T>::NNSystem(const NNSystem<U>& other)
//    : NNSystem<T>(other.get_num_input_ports(), other.get_input_port(0).size()) {}

// The NN inference method.
template <typename T>
void NNSystem<T>::Forward(const Context<T>& context,
                       BasicVector<T>* out) const {
  Eigen::VectorBlock<VectorX<T>> out_vector = out->get_mutable_value();

  // Zeroes the output.
  out_vector.setZero();

  // Sum each input port into the output.
  const BasicVector<T>* input_vector = this->EvalVectorInput(context, 0);
  // Have to convert input to a torch tensor
//  torch::Tensor in = torch::ones({8});
//  torch::Tensor torch_out = neural_network_->forward(in); // TODO: give an actual input here.
  // Have to put this into a basicvector somehow? TODO: Use Eigen here!!
  // *out = ...
}

}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::NNSystem)
