#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include "Eigen/Dense"
#include "Layer.hpp"

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(int kernel_size, int num_filters, std::string name)
        : Layer(name), _kernel_size(kernel_size), _num_filters(num_filters) {
        _kernels = std::vector<Eigen::MatrixXd>(num_filters);
        for (int i = 0; i < num_filters; ++i) {
            _kernels[i] = Eigen::MatrixXd::Random(kernel_size, kernel_size);
        }
        _biases = std::vector<double>(num_filters);
        for (int i = 0; i < num_filters; i++)
        {
            _biases[i] = std::rand() / double(RAND_MAX);
        }
        
    }
    
    /**
     * @brief Performs the forward pass of the convolutional layer.
     *
     * This function applies the convolution operation to the input matrix and
     * caches the input for use in the backward pass.
     *
     * @param input The input matrix to the convolutional layer.
     * @return The output matrix after applying the convolution operation.
     */
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;
    
    /**
     * @brief Performs the backward propagation step for the convolutional layer.
     *
     * This function computes the gradients of the input with respect to the given
     * gradient, and applies the derivative of the convolution operation to update
     * the weights and biases.
     *
     * @param gradient The gradient of the following layer.
     * @param learning_rate The learning rate for weight and bias updates.
     *
     * @return The gradient to be passed to the previous layer.
     */
    Eigen::MatrixXd backward(const Eigen::MatrixXd &gradient,
                             double learning_rate = 1) override;

private:
    int _kernel_size;          // Size of the convolutional kernel
    int _num_filters;          // Number of filters in the convolutional layer
    std::vector<Eigen::MatrixXd> _kernels;  // Weights for the convolutional filters
    std::vector<double> _biases;   // Biases for the convolutional filters
    Eigen::MatrixXd _input_cache; // Cache for the input matrix during the forward pass

#ifdef UNIT_TEST
public:
    Eigen::MatrixXd& getKernels() { return _kernels; }
    Eigen::VectorXd& getBiases() { return _biases; }

    void setKernels(const Eigen::MatrixXd& kernels) { _kernels = kernels; }
    void setBiases(const Eigen::VectorXd& biases) { _biases = biases; }
#endif
};

#endif // CONVOLUTIONAL_LAYER_HPP
