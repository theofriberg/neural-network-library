#include "../include/DenseLayer.hpp"
#include <Eigen/Dense>

#include <iostream>

MatrixXdRM DenseLayer::forward(const MatrixXdRM &input)
{
    _input_cache = input; // Cache the input for use in backward pass
    assert(input.rows() == _weights.cols() && "Input size mismatch");
    MatrixXdRM output = _weights * input;
    output.colwise() += _biases;
    return output;
}

MatrixXdRM DenseLayer::backward(const MatrixXdRM &gradient, double learning_rate)
{
    assert(gradient.rows() == _weights.rows() && "Gradient size mismatch");

    _weights -= learning_rate * (gradient * _input_cache.transpose());
    _biases -= learning_rate * gradient.rowwise().sum();
    return _weights.transpose() * gradient;
}