#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/framework/output_port.h"

namespace drake {
namespace systems {

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
template <typename T>
class MLPSystem final : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MLPSystem) // Sure?

  /// Construct an %MLPSystem System.
  /// @param num_inputs is the number of input neurons.
  /// @param num_hidden is the number of neurons in each hidden layer.
  /// @param num_layers is the number of hidden layers, all of size num_hidden.
  /// @param num_outputs is the number of output neurons.
  MLPSystem(int num_inputs, int num_hidden, int num_layers, int num_outputs);

  /// Scalar-converting copy constructor.  See @ref system_scalar_conversion. // Sure...
  template <typename U>
  explicit MLPSystem(const MLPSystem<U>&);

  /// Returns the output port on which the MLP output is presented.
  const OutputPort<T>& get_output_port() const {
    return LeafSystem<T>::get_output_port(0);
  }

 private:
  // Performs fwd inference of the net, which has user-specified dimensions. If the
  // input ports are not the appropriate count or size, std::runtime_error will
  // be thrown.
  void Forward(const Context<T>& context, BasicVector<T>* sum) const;
};

}  // namespace systems
}  // namespace drake

