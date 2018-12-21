#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/framework/output_port.h"

#include <torch/torch.h>

namespace drake {
namespace systems {

// Declare the format of Torch net objects?
struct DrakeNet : torch::nn::Module {
  virtual torch::Tensor forward(torch::Tensor x) = 0;
};
//struct DrakeNet : torch::nn::Module {
//  virtual int forward() = 0;
//};
//typedef torch::nn::Module DrakeNet;

/// A neural network system that accepts, inputs, parameters, and produces and output.
/// Supports gradients, and is powered by Torchlib, the c++ frontend to PyTorch.
/// @ingroup TODO
///
/// TODO: do I need to specify parameters here?
/// @system{NNSysten,
///    @input_port{input}
///    @output_port{output}
/// }
///
/// TODO: do I need anything here on this line?
/// @tparam T The type of mathematical object being added.
///
/// Instantiated templates for the following kinds of T's are provided:
///
/// - double <- get this = #1
/// - AutoDiffXd <- get this = #2
///
/// TODO: these statements valid?
/// They are already available to link against in the containing library.
/// No other values for T are currently supported.
class NNSystem final : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(NNSystem) // Sure?

  /// Construct an %NNSystem System.
  /// @param neural_network is a TorchLib net, a subclass of torch::nn::Module that has a forward() method.
  NNSystem(DrakeNet *neural_network, int n_inputs, int n_outputs); // TODO: what the hell type do I use here?

  /// Scalar-converting copy constructor.  See @ref system_scalar_conversion. // Sure...
//  template <typename U>
//  explicit NNSystem(const NNSystem<U>&);

  /// Returns the output port on which the NN output is presented.
  const OutputPort<double>& get_output_port() const {
    return LeafSystem<double>::get_output_port(0);
  }

 private:
  // Performs fwd inference of the net, which has user-specified dimensions. If the
  // input ports are not the appropriate count or size, std::runtime_error will
  // be thrown.
  void Forward(const Context<double>& context, BasicVector<double>* sum) const;
  //torch::nn::Module neural_network_;
  DrakeNet *neural_network_; // TODO: switch to a unique pointer so that you have ownership of the network!
  int n_inputs_;
  int n_outputs_;
};

}  // namespace systems
}  // namespace drake

