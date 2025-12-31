#include <Eigen/Dense>

#include "ActivationLayer.hpp"

MatrixXdRM ActivationLayer::forward(const MatrixXdRM &input)
{
    _input_cache = input;
    return _activation_fn(input);
}

MatrixXdRM ActivationLayer::backward(const MatrixXdRM &gradient, double learning_rate)
{
    Eigen::MatrixXd input_prime = _activation_prime(_input_cache);
    return gradient.cwiseProduct(input_prime);
}
