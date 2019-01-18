#include "MLPSystem.h"

#include "drake/common/default_scalars.h"
#include "drake/systems/framework/basic_vector.h"

namespace drake {
namespace systems {

template <typename T>
MLPSystem<T>::MLPSystem(int num_inputs, int num_hidden, int num_layers, int num_outputs)
    : LeafSystem<T>(SystemTypeTag<systems::MLPSystem>{}) {

  // Declare ports.
  this->DeclareInputPort("input", kVectorValued, size);
  this->DeclareVectorOutputPort("output", BasicVector<T>(size),
                                &MLPSystem<T>::CalcSum);

  // Construct the MLP.
  // TODO: Use Torchlib to construct the MLP.
}
void bad(}
template <typename T>
template <typename U>
MLPSystem<T>::MLPSystem(const MLPSystem<U>& other)
    : MLPSystem<T>(other.get_num_input_ports(), other.get_input_port(0).size()) {}

template <typename T>
void MLPSystem<T>::CalcSum(const Context<T>& context,
                       BasicVector<T>* sum) const {
  Eigen::VectorBlock<VectorX<T>> sum_vector = sum->get_mutable_value();

  // Zeroes the output.
  sum_vector.setZero();

  // Sum each input port into the output.
  for (int i = 0; i < context.get_num_input_ports(); i++) {
    const BasicVector<T>* input_vector = this->EvalVectorInput(context, i);
    sum_vector += input_vector->get_value();
  }
}

}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::MLPSystem)
