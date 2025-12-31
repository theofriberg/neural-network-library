#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include "Layer.hpp"
#include "Types.hpp"
#include <Eigen/Dense>

class ConvolutionalLayer : public Layer
{
  public:
    ConvolutionalLayer(int kernel_size, int num_filters, int input_width, int input_height)
        : Layer("Convolutional Layer"), _kernel_size(kernel_size), _num_filters(num_filters),
          _input_width(input_width), _input_height(input_height)
    {
        // Initialize kernels and biases
        _kernels.resize(num_filters);
        _biases.resize(num_filters);
        for (int i = 0; i < num_filters; ++i)
        {
            _kernels[i] = MatrixXdRM::Random(kernel_size, kernel_size);
            _biases[i] = 0.0;
        }
    }

    MatrixXdRM forward(const MatrixXdRM &input) override;
    MatrixXdRM backward(const MatrixXdRM &gradient, double learning_rate = 1) override;

  private:
    MatrixXdRM _input_cache;          // Cache for the input during forward pass
    int _input_width;                 // Width of the input feature map
    int _input_height;                // Height of the input feature map
    int _kernel_size;                 // Size of the convolutional kernel
    int _num_filters;                 // Number of filters in the convolutional layer
    std::vector<MatrixXdRM> _kernels; // Weights for the convolutional filters
    std::vector<double> _biases;      // Biases for the convolutional filters

#ifdef UNIT_TEST
  public:
    std::vector<MatrixXdRM> &getKernels() { return _kernels; }
    std::vector<double> &getBiases() { return _biases; }

    void setKernels(std::vector<MatrixXdRM> &kernels) { _kernels = kernels; }
    void setBiases(std::vector<double> &biases) { _biases = biases; }
#endif
};

#endif // CONVOLUTIONAL_LAYER_HPP
