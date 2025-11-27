#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>

class ActivationLayer : public Layer
{
  public:
    ActivationLayer(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> act_fn,
                    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> act_prime, std::string name)
        : Layer(name), _activation_fn(act_fn), _activation_prime(act_prime)
    {
    }

    /**
     * @brief Performs the forward propagation step for the activation layer.
     * This function applies the activation function to the input matrix and
     * caches the input for use in the backward pass.
     * @param input Vector of input matrices to the layer.
     * @return The output matrix after applying the activation function.
     */
    std::vector<Eigen::MatrixXd>
    forwardBatch(const std::vector<Eigen::MatrixXd> &input_batch) override;

    /**
     * @brief Performs the backward propagation step for the activation layer.
     * This function computes the gradient of the loss with respect to the input
     * by applying the derivative of the activation function to the cached input.
     * @param gradient Vector of gradient matrices from the next layer.
     * @param learning_rate The learning rate for scaling the gradient.
     * @return The gradient matrix with respect to the input of this layer.
     */
    std::vector<Eigen::MatrixXd> backwardBatch(const std::vector<Eigen::MatrixXd> &gradient_batch,
                                               double learning_rate = 1) override;

  private:
    // Activation function and its derivative
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> _activation_fn;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> _activation_prime;
    std::vector<Eigen::MatrixXd> _input_cache; // Cache for the input matrix during the forward pass

    Eigen::MatrixXd singleForward(const Eigen::MatrixXd &input);
    Eigen::MatrixXd singleBackward(const Eigen::MatrixXd &gradient, const Eigen::MatrixXd &input,
                                   double learning_rate);
};

#endif // ACTIVATION_LAYER_HPP