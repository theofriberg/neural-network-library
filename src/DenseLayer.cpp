#include "../include/Eigen/Dense"
#include "../include/DenseLayer.hpp"

#include <iostream>

Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd &input)
{
    _input_cache = input;
    assert(_weights.cols() == input.rows());
    Eigen::MatrixXd output = _weights * input;
    output.colwise() += _biases;
    return output;
}

Eigen::MatrixXd DenseLayer::backward(const Eigen::MatrixXd &gradient, double learning_rate)
{
    _weights -= learning_rate * (gradient * _input_cache.transpose());
    _biases -= learning_rate * gradient.rowwise().sum();
    return _weights.transpose() * gradient;
}