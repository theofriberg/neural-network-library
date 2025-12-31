#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "Layer.hpp"
#include "Types.hpp"
#include <Eigen/Dense>

class ActivationLayer : public Layer
{
  public:
    ActivationLayer(MatrixXdRMValuedFunc act_fn, MatrixXdRMValuedFunc act_prime, std::string name)
        : Layer(name), _activation_fn(act_fn), _activation_prime(act_prime)
    {
    }

    /**
     * @brief Performs the forward propagation step for the activation layer.
     * This function applies the activation function to the input and caches
     * the input for use in the backward pass.
     * @param input_batch Matrix of input data where every col is a single input.
     * @return The output matrix after applying the activation function.
     */
    MatrixXdRM forward(const MatrixXdRM &input_batch) override;

    /**
     * @brief Performs the backward propagation step for the activation layer.
     * This function computes the gradient of the loss with respect to the input
     * of the activation layer using the cached input from the forward pass and
     * the derivative of the activation function.
     * @param gradient_batch Matrix of gradients from the next layer where every col is a single
     * gradient.
     * @param learning_rate The learning rate (not used in activation layers but included for
     * interface consistency).
     * @return A matrix of gradients with respect to the input of this layer.
     */
    MatrixXdRM backward(const MatrixXdRM &gradient_batch, double learning_rate = 1) override;

  private:
    // Activation function and its derivative
    MatrixXdRMValuedFunc _activation_fn;
    MatrixXdRMValuedFunc _activation_prime;
    MatrixXdRM _input_cache; // Cache for the input matrix during the forward pass
};

#endif // ACTIVATION_LAYER_HPP