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

template <typename T>
void NNSystem<T>::Forward(const Context<T>& context,
                       BasicVector<T>* out) const {
    // For non-specialized templates, No-Op.
}

// The NN inference method. Attempting to do them with template
// specializations to handle the gradient propagating case specifically!
template <>
void NNSystem<double>::Forward(const Context<double>& context,
                       BasicVector<double>* out) const {
  // Have to convert input to a torch tensor
  torch::Tensor in = torch::zeros({n_inputs_});
  auto in_a = in.accessor<float,1>();
  const BasicVector<double>* input_vector = this->EvalVectorInput(context, 0);
  for (int i=0; i<n_inputs_; i++){
      in_a[i] = input_vector->GetAtIndex(i);
  }

  // Run the forward pass!
  torch::Tensor torch_out = neural_network_->forward(in);

  // Have to put this into a basicvector somehow? TODO: Use Eigen here?
  auto y_a = torch_out.accessor<float,1>();
  for (int i=0; i<n_outputs_; i++){
      out->SetAtIndex(i, y_a[i]);
  }
}

template <>
void NNSystem<AutoDiffXd>::Forward(const Context<AutoDiffXd>& context,
                       BasicVector<AutoDiffXd>* drake_out) const {

  // Convert input to a torch tensor
  torch::Tensor torch_in = torch::zeros({n_inputs_}, torch::TensorOptions().requires_grad(true));
  auto torch_in_a = torch_in.accessor<float,1>();
  const BasicVector<AutoDiffXd>* drake_in = this->EvalVectorInput(context, 0);
  for (int i=0; i<n_inputs_; i++){
      torch_in_a[i] = drake_in->GetAtIndex(i).value();
  }

  // Run the forward pass.
  // We'll do the backward pass(es) when we calculate output and it's gradients.
  torch::Tensor torch_out = neural_network_->forward(torch_in);

  // Do derivative calculation and pack into the output vector.
  //   Because neural network might have multiple outputs, I can't simply use net.backward() with no argument.
  //   Instead need to follow the advice here:
  //   https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059
  //   auto deriv = Vector1<double>::Constant(1.0);
  auto torch_out_a = torch_out.accessor<float,1>();
  for (int j=0; j<n_outputs_; j++){
      auto y_j_value = torch_out_a[j];
      // Equation: y.derivs = dydu*u.derivs() + dydp*p.derivs()
      // Alternate equation, for each y, y_j.deriv = sum_i  (dy_jdu[i] * u[i].deriv)   
      //                        (#y's)  (1x#derivs)  (#u's) (1x#u's)     (1x#derivs)
      
      // Make empty accumulator
      auto y_j_deriv = drake_in->GetAtIndex(0).derivatives(); // TODO ensure that this does not copy!
      y_j_deriv = y_j_deriv.Zero(y_j_deriv.size());

      //   https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059
      auto output_selector = torch::zeros({1, n_outputs_});
      output_selector[j] = 1.0; // Set the output we want a derivative w.r.t. to 1.
      torch_out.backward(output_selector, /*keep_graph*/true);
      auto dy_jdu = torch_in.grad(); // From Torch
      auto dy_jdu_a = dy_jdu.accessor<float,1>(); // From Torch
      for (int i=0; i<n_inputs_; i++){
          auto u_i_deriv = drake_in->GetAtIndex(i).derivatives();
          std::cout << "dy_jdu_a[i] * u_i_deriv = " << dy_jdu_a[i] << " * " <<  u_i_deriv << std::endl;
          y_j_deriv += dy_jdu_a[i] * u_i_deriv;
      }
      std::cout << "putting into output: " << y_j_value << ", " << y_j_deriv << std::endl;
      drake_out->SetAtIndex(j, AutoDiffXd(y_j_value, y_j_deriv));
  }
}

}  // namespace systems
}  // namespace drake

// Should eventually use the non-symbolic version here!
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::NNSystem)
