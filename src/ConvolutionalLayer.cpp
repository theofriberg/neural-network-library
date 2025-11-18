#include "../include/ConvolutionalLayer.hpp"
#include "../include/CorrelationCalculator.hpp"

std::vector<Eigen::MatrixXd> ConvolutionalLayer::forwardBatch(const std::vector<Eigen::MatrixXd> &input) {
    _input_cache = input;
    std::vector<Eigen::MatrixXd> output_batch;
    output_batch.reserve(input.size());
    for (const auto& input_matrix : input) {
        output_batch.push_back(singleForward(input_matrix));
    }
    return output_batch;
}

std::vector<Eigen::MatrixXd> ConvolutionalLayer::backwardBatch(const std::vector<Eigen::MatrixXd> &gradient_batch,
                                 double learning_rate) {
    std::vector<Eigen::MatrixXd> input_gradient_batch;
    input_gradient_batch.reserve(gradient_batch.size());
    for (int i = 0; i < gradient_batch.size(); i++)
    {
        input_gradient_batch.push_back(
            singleBackward(gradient_batch[i], _input_cache[i], learning_rate)
        );
    }
    
    return input_gradient_batch;
}

Eigen::MatrixXd ConvolutionalLayer::singleForward(const Eigen::MatrixXd &input) {
    int input_rows = input.rows();
    int input_cols = input.cols();

    int output_rows = input_rows + _kernel_size - 1;
    int output_cols = input_cols + _kernel_size - 1;

    Eigen::MatrixXd output(output_rows, output_cols * _num_filters);

    // Perform convolution operation
    for (int f = 0; f < _num_filters; ++f) {
        output.block(0, f * output_cols, output_rows, output_cols) =
            CorrelationCalculator::convolve(input, _kernels[f]);
    }

    return output;
}

Eigen::MatrixXd ConvolutionalLayer::singleBackward(const Eigen::MatrixXd &gradient, const Eigen::MatrixXd &input, double learning_rate) {
    int input_rows = input.rows();
    int input_cols = input.cols();

    Eigen::MatrixXd input_gradient = Eigen::MatrixXd::Zero(input_rows, input_cols);

    int output_cols = gradient.cols() / _num_filters;

    for (int f = 0; f < _num_filters; ++f) {
        Eigen::MatrixXd grad_slice = gradient.block(0, f * output_cols, gradient.rows(), output_cols);
        input_gradient += CorrelationCalculator::convolve(grad_slice, _kernels[f]);
        _kernels[f] -= learning_rate * CorrelationCalculator::convolve(input, grad_slice);;
        _biases[f] -= learning_rate * grad_slice.sum();
    }

    return input_gradient;
}