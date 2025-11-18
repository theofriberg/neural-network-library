#include <Eigen/Dense>
#include "../include/DenseLayer.hpp"

#include <iostream>

std::vector<Eigen::MatrixXd> DenseLayer::forwardBatch(const std::vector<Eigen::MatrixXd> &input_batch)
{
    _input_cache = input_batch;
    std::vector<Eigen::MatrixXd> output_batch;
    for (const auto &input : input_batch) {
        output_batch.push_back(singleForward(input));
    }
    return output_batch;
}

std::vector<Eigen::MatrixXd> DenseLayer::backwardBatch(const std::vector<Eigen::MatrixXd> &gradient_batch,
                                   double learning_rate)
{
    std::vector<Eigen::MatrixXd> input_gradient_batch;
    for (int i = 0; i < gradient_batch.size(); i++)
    {
        input_gradient_batch.push_back(singleBackward(gradient_batch[i], _input_cache[i], learning_rate));
    }
    
    return input_gradient_batch;
}

Eigen::MatrixXd DenseLayer::singleForward(const Eigen::MatrixXd &input)
{
    assert(input.cols() == 1 && "DenseLayer expects a column vector");
    assert(input.rows() == _weights.cols() && "Input vector size mismatch");
    
    Eigen::MatrixXd output = _weights * input;
    output.colwise() += _biases;
    return output;
}

Eigen::MatrixXd DenseLayer::singleBackward(const Eigen::MatrixXd &gradient, const Eigen::MatrixXd &input, double learning_rate)
{
    assert(gradient.rows() == _weights.rows() && "Gradient size mismatch");
    assert(gradient.cols() == 1 && "DenseLayer expects gradient as column vector");

    _weights -= learning_rate * (gradient * input.transpose());
    _biases -= learning_rate * gradient.rowwise().sum();
    return _weights.transpose() * gradient;
}