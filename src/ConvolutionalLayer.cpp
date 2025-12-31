#include "ConvolutionalLayer.hpp"
#include "CorrelationCalculator.hpp"

MatrixXdRM ConvolutionalLayer::forward(const MatrixXdRM &input)
{
    _input_cache = input;
    MatrixXdRM output(_num_filters * _input_width * _input_height, input.cols());
    for (int b = 0; b < input.cols(); ++b)
    {
        MatrixXdRM single_input = input.col(b).reshaped(_input_height, _input_width);
        MatrixXdRM output_feature_map(_num_filters, _input_width * _input_height);
        for (int f = 0; f < _num_filters; ++f)
        {
            MatrixXdRM convolved = CorrelationCalculator::convolve(single_input, _kernels[f]);
            assert(convolved.rows() == _input_height && convolved.cols() == _input_width &&
                   "Convolved output has incorrect dimensions.");
            convolved.array() += _biases[f];
            output_feature_map.row(f) =
                Eigen::Map<Eigen::RowVectorXd>(convolved.data(), convolved.size());
        }
        output.col(b) = Eigen::Map<VectorXd>(output_feature_map.data(), output_feature_map.size());
    }

    return output;
}

MatrixXdRM ConvolutionalLayer::backward(const MatrixXdRM &gradient, double learning_rate)
{
    MatrixXdRM input_gradient = MatrixXdRM::Zero(_input_width * _input_height, gradient.cols());
    std::vector<MatrixXdRM> kernel_gradients(_num_filters,
                                             MatrixXdRM::Zero(_kernel_size, _kernel_size));
    std::vector<double> bias_gradients(_num_filters, 0.0);

    for (int b = 0; b < gradient.cols(); ++b)
    {
        MatrixXdRM single_input = _input_cache.col(b).reshaped(_input_height, _input_width);
        MatrixXdRM single_gradient =
            gradient.col(b).reshaped(_num_filters, _input_width * _input_height);

        for (int f = 0; f < _num_filters; ++f)
        {
            MatrixXdRM grad_feature_map =
                Eigen::Map<MatrixXdRM>(single_gradient.row(f).data(), _input_height, _input_width);

            // Compute input gradient
            MatrixXdRM input_grad_part =
                CorrelationCalculator::convolve(grad_feature_map, _kernels[f]);
            input_gradient.col(b) +=
                Eigen::Map<VectorXd>(input_grad_part.data(), input_grad_part.size());

            // Accumulate kernel and bias gradients
            MatrixXdRM full_grad =
                CorrelationCalculator::crossCorrelate(single_input, grad_feature_map);

            // Skär ut den centrala _kernel_size x _kernel_size delen
            int start_row = (full_grad.rows() - _kernel_size) / 2;
            int start_col = (full_grad.cols() - _kernel_size) / 2;
            MatrixXdRM grad_kernel =
                full_grad.block(start_row, start_col, _kernel_size, _kernel_size);
            kernel_gradients[f] += grad_kernel;

            bias_gradients[f] += grad_feature_map.sum();
        }
    }
    // Update kernels and biases
    for (int f = 0; f < _num_filters; ++f)
    {
        int batch_size = _input_cache.cols();
        _kernels[f] -= learning_rate * kernel_gradients[f] / batch_size;
        _biases[f] -= learning_rate * bias_gradients[f] / batch_size;
    }

    return input_gradient;
}