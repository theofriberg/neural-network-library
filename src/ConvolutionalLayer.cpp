#include "../include/ConvolutionalLayer.hpp"
#include "../include/CorrelationCalculator.hpp"

Eigen::MatrixXd ConvolutionalLayer::forward(const Eigen::MatrixXd &input) {
    _input_cache = input;

    int input_rows = input.rows();
    int input_cols = input.cols();

    int output_rows = input_rows - _kernel_size + 1;
    int output_cols = input_cols - _kernel_size + 1;

    Eigen::MatrixXd output(output_rows, output_cols * _num_filters);

    // Perform convolution operation
    for (int f = 0; f < _num_filters; ++f) {
        output.block(0, f * output_cols, output_rows, output_cols) =
            CorrelationCalculator::convolve(const_cast<Eigen::MatrixXd&>(input), _kernels[f]);
    }

    return output;
}

Eigen::MatrixXd ConvolutionalLayer::backward(const Eigen::MatrixXd &gradient, double learning_rate) {
    int input_rows = _input_cache.rows();
    int input_cols = _input_cache.cols();

    Eigen::MatrixXd input_gradient = Eigen::MatrixXd::Zero(input_rows, input_cols);

    int output_cols = gradient.cols() / _num_filters;

    for (int f = 0; f < _num_filters; ++f) {
        Eigen::MatrixXd grad_slice = gradient.block(0, f * output_cols, gradient.rows(), output_cols);
        input_gradient += CorrelationCalculator::convolve(grad_slice, _kernels[f]);
        _kernels[f] -= learning_rate * CorrelationCalculator::convolve(_input_cache, grad_slice);;
        _biases[f] -= learning_rate * grad_slice.sum();
    }

    return input_gradient;
}