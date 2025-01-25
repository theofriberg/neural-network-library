#include <Eigen/Dense>
#include "Layer.hpp"

class DenseLayer : public Layer
{
public:
    /**
     * @brief Constructor for the DenseLayer class.
     *
     * Initializes the dense layer with random weights and biases.
     *
     * @param input_size The number of input features to the dense layer.
     * @param output_size The number of output features produced by the dense layer.
     *
     * @note The weights and biases are initialized using Eigen's MatrixXd::Random and
     *       VectorXd::Random functions, respectively.
     */
    DenseLayer(int input_size, int output_size)
        : _weights(Eigen::MatrixXd::Random(output_size, input_size)),
          _biases(Eigen::VectorXd::Random(output_size)) {}

    /**
     * @brief Performs a forward pass through the dense layer.
     *
     * This function takes an input matrix and applies the dense layer's weights and biases
     * to produce an output matrix. The forward pass is used during the training process
     * to propagate the input data through the neural network.
     *
     * @param input The input matrix to the dense layer. It should have dimensions
     *              (input_size, batch_size), where input_size is the number of input
     *              features and batch_size is the number of samples in the input.
     *
     * @return The output matrix after applying the dense layer's weights and biases.
     *         It has dimensions (output_size, batch_size), where output_size is the
     *         number of output features.
     */
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

    Eigen::MatrixXd backward(const Eigen::MatrixXd &gradient) override;

private:
    Eigen::MatrixXd _weights;
    Eigen::VectorXd _biases;
};