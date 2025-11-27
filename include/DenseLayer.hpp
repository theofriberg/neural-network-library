#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>

class DenseLayer : public Layer
{
  public:
    /**
     * @brief Constructor for the DenseLayer class.
     *
     * Initializes the dense layer with random weights and biases.
     *
     * @param input_size The number of input features to the dense layer.
     * @param output_size The number of output features produced by the dense
     * layer.
     *
     * @note The weights and biases are initialized using Eigen's MatrixXd::Random
     * and VectorXd::Random functions, respectively.
     */
    DenseLayer(int input_size, int output_size)
        : Layer("Dense"), _weights(Eigen::MatrixXd::Random(output_size, input_size)),
          _biases(Eigen::VectorXd::Random(output_size))
    {
    }

    /**
     * @brief Performs the forward pass of the dense layer.
     * This function computes the output of the dense layer by performing a
     * matrix multiplication of the input with the weights and adding the biases.
     * It also caches the input for use in the backward pass.
     * @param input_batch A vector of input matrices to the dense layer.
     * @return A vector of output matrices after applying the dense layer.
     */
    std::vector<Eigen::MatrixXd>
    forwardBatch(const std::vector<Eigen::MatrixXd> &input_batch) override;

    /**
     * @brief Performs the backward propagation step for the dense layer.
     * This function computes the gradients of the input with respect to the
     * given gradient, and applies the derivative of the dense layer to update
     * the weights and biases.
     * @param gradient_batch A vector of gradients from the following layer.
     * @param learning_rate The learning rate for weight and bias updates.
     * @return A vector of gradients to be passed to the previous layer.
     */
    std::vector<Eigen::MatrixXd> backwardBatch(const std::vector<Eigen::MatrixXd> &gradient_batch,
                                               double learning_rate = 1) override;

    Eigen::MatrixXd getWeights() const { return _weights; }
    Eigen::VectorXd getBiases() const { return _biases; }

  private:
    Eigen::MatrixXd _weights;
    Eigen::VectorXd _biases;
    std::vector<Eigen::MatrixXd> _input_cache; // Cache for the input matrix during the forward pass

    Eigen::MatrixXd singleForward(const Eigen::MatrixXd &input);
    Eigen::MatrixXd singleBackward(const Eigen::MatrixXd &gradient, const Eigen::MatrixXd &input,
                                   double learning_rate);
};

#endif // DENSE_LAYER_HPP