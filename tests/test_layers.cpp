#include <gtest/gtest.h>

#include "../include/Eigen/Dense"
#include "../include/DenseLayer.hpp"
#include "../include/ActivationLayers.hpp"

#include <iostream>

class DenseLayerTest : public ::testing::Test
{
protected:
    // Set up a DenseLayer with 3 inputs and 2 outputs (arbitrary values for testing)
    DenseLayer *layer;

    void SetUp() override
    {
        layer = new DenseLayer(3, 2);
    }

    void TearDown() override
    {
        delete layer;
    }
};

// Test forward pass
TEST_F(DenseLayerTest, ForwardPass)
{
    // Input matrix with 3 features and 2 samples (3x2)
    Eigen::MatrixXd input(3, 2);
    input << 1, 2,
        3, 4,
        5, 6;

    // Call forward pass
    Eigen::MatrixXd output = layer->forward(input);

    // Check the output dimensions (should be 2x2, since the output size is 2)
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 2);

    // Check that the output is computed as weights * input + biases
    // This is more of an indirect verification that the output looks reasonable
    // since it's hard to compare exact values due to random weights and biases.
    EXPECT_FALSE(output.isZero()); // Just a basic check for non-zero output
}

// Test backward pass
TEST_F(DenseLayerTest, BackwardPass)
{
    // Input matrix with 3 features and 2 samples (3x2)
    Eigen::MatrixXd input(3, 2);
    input << 1, 2,
        3, 4,
        5, 6;

    // Create a dummy gradient matrix (2x2, matching output size)
    Eigen::MatrixXd gradient(2, 2);
    gradient << 0.1, 0.2,
        0.3, 0.4;

    // Get initial weights and biases before backward pass
    Eigen::MatrixXd initial_weights = layer->getWeights();
    Eigen::VectorXd initial_biases = layer->getBiases();

    // Perform forward pass
    Eigen::MatrixXd _ = layer->forward(input);

    // Perform backward pass
    Eigen::MatrixXd prev_layer_gradient = layer->backward(gradient, 0.01);

    // Check that the previous layer gradient has the correct dimensions (should match input size)
    ASSERT_EQ(prev_layer_gradient.rows(), 3);
    ASSERT_EQ(prev_layer_gradient.cols(), 2);

    Eigen::MatrixXd updated_weights = layer->getWeights();
    Eigen::VectorXd updated_biases = layer->getBiases();

    ASSERT_FALSE(updated_weights.isApprox(initial_weights)); // Check if weights have changed
    ASSERT_FALSE(updated_biases.isApprox(initial_biases));   // Check if biases have changed
}

// Test layer with random data
TEST_F(DenseLayerTest, RandomDataTest)
{
    // Create random input data (3x2)
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(3, 2);

    // Perform forward pass
    Eigen::MatrixXd output = layer->forward(input);

    // Ensure that the output is not zero (as it involves random weights and biases)
    EXPECT_FALSE(output.isZero());

    // Ensure that the output dimensions are correct (should be 2x2)
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 2);
}

// Test layer with zero gradients during backward pass
TEST_F(DenseLayerTest, ZeroGradientTest)
{
    // Create random input data (3x2)
    Eigen::MatrixXd input(3, 2);
    input << 1, 2,
        3, 4,
        5, 6;

    // Create zero gradient (2x2, same size as the output)
    Eigen::MatrixXd zero_gradient(2, 2);
    zero_gradient.setZero();

    // Get initial weights and biases before backward pass
    Eigen::MatrixXd initial_weights = layer->getWeights();
    Eigen::VectorXd initial_biases = layer->getBiases();

    // Perform forward pass
    Eigen::MatrixXd _ = layer->forward(input);

    // Perform backward pass
    Eigen::MatrixXd prev_layer_gradient = layer->backward(zero_gradient, 0.01);

    // Ensure the previous layer gradient is also zero
    ASSERT_TRUE(prev_layer_gradient.isZero());

    // Ensure that the weights and biases remain the same after a zero gradient update
    Eigen::MatrixXd updated_weights = layer->getWeights();
    Eigen::VectorXd updated_biases = layer->getBiases();

    EXPECT_TRUE(initial_weights.isApprox(updated_weights));
    EXPECT_TRUE(initial_biases.isApprox(updated_biases));
}

class ActivationLayerTest : public ::testing::Test
{
protected:
    // Set up a ReLU activation layer
    ReLU *relu_layer;

    void SetUp() override
    {
        relu_layer = new ReLU();
    }

    void TearDown() override
    {
        delete relu_layer;
    }
};

TEST_F(ActivationLayerTest, ReLUForward)
{
    Eigen::MatrixXd input(2, 2);
    input << -1, 2,
        0, -3;

    Eigen::MatrixXd expected(2, 2);
    expected << 0, 2,
        0, 0;

    // Ensure forward gives expected output
    Eigen::MatrixXd output = relu_layer->forward(input);
    EXPECT_TRUE(output.isApprox(expected));
}

TEST_F(ActivationLayerTest, ReLUBackward)
{
    Eigen::MatrixXd input(2, 2);
    input << -1, 2,
        0, -3;

    Eigen::MatrixXd gradient(2, 2);
    gradient << 0.1, 0.2,
        0.3, 0.4;

    Eigen::MatrixXd expected_gradient(2, 2);
    expected_gradient << 0, 0.002,
        0, 0;

    // Perform a forward pass
    Eigen::MatrixXd _ = relu_layer->forward(input);

    // Ensure backward gives expected gradient
    Eigen::MatrixXd relu_gradient = relu_layer->backward(gradient, 0.01);
    EXPECT_TRUE(relu_gradient.isApprox(expected_gradient));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}