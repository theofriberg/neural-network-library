#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "Eigen/Dense"
#include "Layer.hpp"

class ActivationLayer : public Layer
{
public:
    ActivationLayer(
        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> act_fn,
        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> act_prime) : _activation_fn(act_fn),
                                                                     _activation_prime(act_prime) {}

    /**
     * @brief Performs the forward pass of the activation layer.
     *
     * This function applies the activation function to the input matrix and caches the input for
     * use in the backward pass.
     *
     * @param input The input matrix to the activation layer.
     * @return The output matrix after applying the activation function.
     */
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

    /**
     * @brief Performs the backward propagation step for the activation layer.
     *
     * This function computes the gradients of the input with respect to the given gradient,
     * and applies the derivative of the activation function to update the input weights.
     *
     * @param gradient The gradient of the following layer.
     * @param learning_rate The learning rate for weight and bias updates.
     *
     * @return The gradient to be passed to the previous layer.
     */
    Eigen::MatrixXd backward(const Eigen::MatrixXd &gradient, double learning_rate = 1) override;

private:
    // Activation function and its derivative
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> _activation_fn;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> _activation_prime;
    Eigen::MatrixXd _input_cache; // Cache for the input matrix during the forward pass
};

#endif // ACTIVATION_LAYER_HPP