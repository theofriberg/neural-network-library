#include <iostream>
#include "../include/ReshapeLayer.hpp"

std::vector<Eigen::MatrixXd> ReshapeLayer::forwardBatch(const std::vector<Eigen::MatrixXd> &input) {
    assert(!input.empty() && "Input batch is empty");
    _input_rows = input[0].rows();
    _input_cols = input[0].cols();
    std::vector<Eigen::MatrixXd> output_batch;
    for (const auto &matrix : input) {
        assert(matrix.rows() == _input_rows && matrix.cols() == _input_cols && "Input matrix dimensions do not match expected dimensions.");
        Eigen::MatrixXd reshape = matrix.reshaped(_output_rows, _output_cols);
        output_batch.push_back(reshape);
    }
    return output_batch;
}

std::vector<Eigen::MatrixXd> ReshapeLayer::backwardBatch(const std::vector<Eigen::MatrixXd> &gradient_batch, 
                                                         double learning_rate) {
    std::vector<Eigen::MatrixXd> output_batch;
    for (const auto &matrix : gradient_batch) {
        assert(matrix.size() == _input_rows * _input_cols && "Gradient size does not match input size in ReshapeLayer backward");
        Eigen::MatrixXd reshape = matrix.reshaped(_input_rows, _input_cols);
        output_batch.push_back(reshape);
    }
    return output_batch;
}