#include "NNSystem.h"

#include "drake/common/default_scalars.h"
#include "drake/systems/framework/basic_vector.h"

#include <torch/torch.h>

namespace drake {
namespace systems {

// Currently hardcoded to only support vector in and vector out.
// User needs to specify how many input and output units.
template <typename T>
NNSystem<T>::NNSystem(DrakeNet *neural_network, bool declare_params)
    : LeafSystem<T>(SystemTypeTag<NNSystem>{}),
      neural_network_(neural_network),
      declare_params_(declare_params) {

    // Determine our neural network dimensions.
    // Right now, because we can't switch out the network, these dimensions
    // will be valid for the lifetime of NNSystem.
    auto parameters = neural_network_->parameters();
    auto first_layer_sizes = parameters[0].sizes();
    auto last_layer_sizes = parameters[parameters.size() - 1].sizes();

    // For some reason, the layer sizes are the transposes of the weight matrices.
    n_inputs_  = first_layer_sizes[1];
    n_outputs_ = last_layer_sizes[0];

    // Get number of parameters.
    if (declare_params_) {
        n_params_ = 0;
        for (auto parameter : parameters){
            // For now, only support two dimensional weight matrices.
            DRAKE_DEMAND( parameter.sizes().size() <= 2);
            auto params_flat = parameter.flatten();
            n_params_ += params_flat.size(0);
        }

        // Make a parameter vector for our Context.
        int params_loaded = 0;
        params_ = std::make_unique<BasicVector<T>>(n_params_);
        for (auto parameter : parameters){
            auto params_flat = parameter.flatten();
            auto params_flat_a = params_flat.accessor<float, 1>();
            int n = params_flat.size(0);
            for (int i=0; i<n; i++)
                params_->SetAtIndex(params_loaded + i,  static_cast<T>(params_flat_a[i]));
            params_loaded += n;
        }
        DRAKE_DEMAND(params_loaded == n_params_);
    }

    // Declare ports.
    this->DeclareInputPort("NN_in", kVectorValued, n_inputs_);
    this->DeclareVectorOutputPort("NN_out", BasicVector<T>(n_outputs_),
                                  &NNSystem::Forward);
    if (declare_params_)
        this->DeclareNumericParameter(*params_);
}

template <typename T>
template <typename U>
NNSystem<T>::NNSystem(const NNSystem<U>& other)
    : NNSystem<T>(other.get_neural_network()) {}

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
    // Drake input -> Torch input
    torch::Tensor in = torch::zeros({n_inputs_});
    auto in_a = in.accessor<float,1>();
    const BasicVector<double>* input_vector = this->EvalVectorInput(context, 0);
    for (int i=0; i<n_inputs_; i++){
        in_a[i] = input_vector->GetAtIndex(i);
    }
  
    // Forward pass: Torch input -> Torch output.
    torch::Tensor torch_out = neural_network_->forward(in);
  
    // Torch output -> Drake output.
    auto y_a = torch_out.accessor<float,1>();
    for (int i=0; i<n_outputs_; i++){
        out->SetAtIndex(i, y_a[i]);
    }
}


template <>
void NNSystem<AutoDiffXd>::Forward(const Context<AutoDiffXd>& context,
                                   BasicVector<AutoDiffXd>* drake_out) const {
  
    // Convert input to a torch tensor
    torch::Tensor torch_in = torch::zeros({n_inputs_}, 
        torch::TensorOptions().requires_grad(true));
    auto torch_in_a = torch_in.accessor<float,1>();
    const BasicVector<AutoDiffXd>* drake_in = this->EvalVectorInput(context, 0);
    for (int i=0; i<n_inputs_; i++){
        torch_in_a[i] = drake_in->GetAtIndex(i).value();
    }
  
    // Run the forward pass.
    // We'll do the backward pass(es) when we calculate output and it's gradients.
    torch::Tensor torch_out = neural_network_->forward(torch_in);
    const int n_derivs  = drake_in->GetAtIndex(0).derivatives().size();
  
    // Make a jacobian of the (n_inputs x n_derivs) in_vec
    Eigen::MatrixXd in_deriv_jac(n_inputs_, n_derivs);
    for (int i=0; i<n_inputs_; i++)
        in_deriv_jac.row(i) = drake_in->GetAtIndex(i).derivatives();

    // Make a jacobian of the (n_param x n_derivs in param_vec)
    Eigen::MatrixXd param_deriv_jac(2, 2);
    if (declare_params_){
        param_deriv_jac.resize(n_params_, n_derivs);
        for (int i=0; i<n_params_; i++)
            param_deriv_jac.row(i) = params_->GetAtIndex(i).derivatives();
    }

    // Make an empty accumulator for Neural network (n_outputs_ vs n_inputs_) jacobian
    Eigen::MatrixXd out_in_jac(n_outputs_, n_inputs_);

    // Make an empty accumulator for Neural network (n_outputs_ vs n_params_) jacobian
    Eigen::MatrixXd out_param_jac(n_outputs_, n_inputs_);

    // Do derivative calculation and pack into the output vector.
    //   Because neural network might have multiple outputs, I can't simply use net.backward() with no argument.
    //   Instead need to follow the advice here:
    //   https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059
    auto torch_out_a = torch_out.accessor<float,1>();
    for (int j=0; j<n_outputs_; j++){
        // Clear out all the existing grads
        neural_network_->zero_grad();

        // Since we are using retain_graph to keep non-leaf gradients, and we 
        // are in a for loop, leaf nodes grad will initially be None, and in subsequent
        // iterations, will be not None.
        if (torch_in.grad().defined())
            torch_in.grad().zero_();

        // https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059
        auto output_selector = torch::zeros({1, n_outputs_});
        output_selector[j] = 1.0; // Set the output we want a derivative w.r.t. to 1.
        torch_out.backward(output_selector, /*keep_graph*/true);

        // Calculate the contribution to y_j.derivs w.r.t all the inputs.
        auto dy_jdu = torch_in.grad(); // From Torch
        auto dy_jdu_a = dy_jdu.accessor<float, 1>(); // From Torch
        for (int i=0; i<n_inputs_; i++)
            out_in_jac(j, i) = dy_jdu_a[i];

        // Optionally add contribution of parameter gradients.
        if (declare_params_) {
            int grads_loaded = 0;
            for (auto parameter : neural_network_->parameters()){
                auto grads_flat = parameter.grad().flatten();
                int n = grads_flat.size(0);
                auto grads_flat_a = grads_flat.accessor<float, 1>();
                for (int i=0; i<n; i++)
                    out_param_jac(j, grads_loaded + i) = grads_flat_a[i];
                grads_loaded += n;
            }
            DRAKE_DEMAND( grads_loaded == n_params_);
        }
    }

    // Apply chain rule to get all derivatives from inputs -> outputs and (optionally) params -> outputs.
    Eigen::MatrixXd out_deriv_jac(n_outputs_, n_derivs);
    out_deriv_jac = out_in_jac * in_deriv_jac;
    std::cout << "out_deriv_jac: " << out_deriv_jac << "out_in_jac: " << out_in_jac << "in_deriv_jac: " << in_deriv_jac << std::endl;

    if (declare_params_)
        out_deriv_jac += out_param_jac * param_deriv_jac;
    std::cout << "out_deriv_jac: " << out_deriv_jac << "out_param_jac: " << out_param_jac << "param_deriv_jac: " << param_deriv_jac << std::endl;

    for (int j=0; j<n_outputs_; j++){
       drake_out->SetAtIndex(j, AutoDiffXd(torch_out_a[j], out_deriv_jac.row(j)));
    }
}

}  // namespace systems
}  // namespace drake

// Should eventually use the non-symbolic version here!
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::systems::NNSystem)

