#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>

class DenseLayer : public Layer
{
  public:
    /**
     * @brief Constructor for the DenseLayer class.
     * Initializes the weights and biases of the dense layer with random values.
     * @param input_size The size of the input vector.
     * @param output_size The size of the output vector.
     */
    DenseLayer(int input_size, int output_size)
        : Layer("Dense"), _weights(MatrixXdRM::Random(output_size, input_size)),
          _biases(VectorXd::Random(output_size))
    {
    }

    /**
     * @brief Performs the forward pass of the dense layer.
     * This function computes the output of the dense layer by performing a
     * matrix multiplication of the input with the weights and adding the biases.
     * It also caches the input for use in the backward pass.
     * @param input_batch Matrix of input data where every col is a single input.
     * @return A vector of output matrices after applying the dense layer.
     */
    MatrixXdRM forward(const MatrixXdRM &input_batch) override;

    /**
     * @brief Performs the backward pass of the dense layer.
     * This function computes the gradient of the loss with respect to the input,
     * weights, and biases of the dense layer. It also updates the weights and
     * biases using the computed gradients and the specified learning rate.
     * @param gradient_batch Matrix of gradients from the next layer where every col is a single
     * gradient.
     * @param learning_rate The learning rate for updating weights and biases.
     * @return A matrix of gradients with respect to the input of this layer.
     */
    MatrixXdRM backward(const MatrixXdRM &gradient_batch, double learning_rate = 1) override;

    MatrixXdRM getWeights() const { return _weights; }
    VectorXd getBiases() const { return _biases; }

  private:
    MatrixXdRM _input_cache; // Cache for the input matrix during the forward pass
    MatrixXdRM _weights;
    VectorXd _biases;
};

#endif // DENSE_LAYER_HPP