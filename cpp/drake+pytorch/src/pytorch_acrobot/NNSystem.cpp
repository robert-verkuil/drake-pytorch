#include "NNSystem.h"

#include "drake/common/default_scalars.h"
#include "drake/systems/framework/basic_vector.h"

#include <torch/torch.h>

namespace drake {
namespace systems {

// Currently hardcoded to only support vector in and vector out.
// User needs to specify how many input and output units.
template <typename T>
NNSystem<T>::NNSystem(DrakeNet *neural_network, int n_inputs, int n_outputs)
    //: LeafSystem<double>(SystemTypeTag<systems::NNSystem>{}), // Will need to enable this again!
    : LeafSystem<T>(SystemTypeTag<NNSystem>{}),
      neural_network_(neural_network),
      n_inputs_(n_inputs),
      n_outputs_(n_outputs) {

  // TODO: investigate determining input and output shape dynamically
  // https://github.com/pytorch/pytorch/blob/eb5d28ecefb9d78d4fff5fac099e70e5eb3fbe2e/torch/csrc/api/include/torch/nn/modules/any.h
  // auto modules = neural_network_->modules(false);

  // Declare ports.
  this->DeclareInputPort("NN_in", kVectorValued, n_inputs); // TODO figure out num_inputs automatically
  this->DeclareVectorOutputPort("NN_out", BasicVector<T>(n_outputs), // TODO figure out num_outputs automatically
                                &NNSystem::Forward);
}

template <typename T>
template <typename U>
NNSystem<T>::NNSystem(const NNSystem<U>& other)
    : NNSystem<T>(other.get_neural_network(), other.get_n_inputs(), other.get_n_outputs()) {}

// The NN inference method.
template <typename T>
void NNSystem<T>::Forward(const Context<T>& context,
                       BasicVector<T>* out) const {
  // Have to convert input to a torch tensor
  torch::Tensor in = torch::zeros({n_inputs_});
  auto in_a = in.accessor<float,1>();
  const BasicVector<T>* input_vector = this->EvalVectorInput(context, 0);
//  for (int i=0; i<n_inputs_; i++){
//      in_a[i] = input_vector->GetAtIndex(i);
//  }

  // Run the forward pass!
  torch::Tensor torch_out = neural_network_->forward(in);

  // Have to put this into a basicvector somehow? TODO: Use Eigen here?
  auto out_a = torch_out.accessor<float,1>();
//  for (int i=0; i<n_outputs_; i++){
//      out->SetAtIndex(i, out_a[i]); // TODO: will this the non-const version? - probably??
//  }
}

}  // namespace systems
}  // namespace drake

// Should eventually use the non-symbolic version here!
//DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
//    class ::drake::systems::NNSystem)
