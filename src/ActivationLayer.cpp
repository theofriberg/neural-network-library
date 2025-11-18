#include <Eigen/Dense>

#include "../include/ActivationLayer.hpp"
#include <iostream>

std::vector<Eigen::MatrixXd> ActivationLayer::forwardBatch(const std::vector<Eigen::MatrixXd> &input_batch)
{
    _input_cache = input_batch;
    std::vector<Eigen::MatrixXd> output_batch;
    for (const auto &input : input_batch) {
        output_batch.push_back(singleForward(input));
    }
    return output_batch;
}

std::vector<Eigen::MatrixXd> ActivationLayer::backwardBatch(const std::vector<Eigen::MatrixXd> &gradient_batch,
                                   double learning_rate)
{
    std::vector<Eigen::MatrixXd> input_gradient_batch;
    for (int i = 0; i < gradient_batch.size(); i++)
    {
        input_gradient_batch.push_back(singleBackward(gradient_batch[i], _input_cache[i], learning_rate));
    }
    return input_gradient_batch;
}

Eigen::MatrixXd ActivationLayer::singleForward(const Eigen::MatrixXd &input)
{
    return _activation_fn(input);
}

Eigen::MatrixXd ActivationLayer::singleBackward(const Eigen::MatrixXd &gradient, const Eigen::MatrixXd &input, double learning_rate)
{
    Eigen::MatrixXd input_prime = _activation_prime(input);
    return learning_rate * (gradient.cwiseProduct(input_prime));
}
