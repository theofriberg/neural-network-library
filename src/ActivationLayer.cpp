#include "../include/Eigen/Dense"

#include "../include/ActivationLayer.hpp"
#include <iostream>

Eigen::MatrixXd ActivationLayer::forward(const Eigen::MatrixXd &input)
{
    _input_cache = input;
    return _activation_fn(input);
}

Eigen::MatrixXd ActivationLayer::backward(const Eigen::MatrixXd &gradient, double learning_rate)
{
    Eigen::MatrixXd input_prime = _activation_prime(_input_cache);
    return learning_rate * (gradient.cwiseProduct(input_prime));
}
