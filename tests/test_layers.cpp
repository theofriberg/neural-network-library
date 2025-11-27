#include <gtest/gtest.h>
#define UNIT_TEST

#include <Eigen/Dense>
#include <iostream>
#include "../include/DenseLayer.hpp"
#include "../include/ActivationLayers.hpp"
#include "../include/ConvolutionalLayer.hpp"
#include "../include/ReshapeLayer.hpp"


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
    Eigen::MatrixXd input(3, 1);
    input << 1, 2, 3;

    // Call forward pass
    Eigen::MatrixXd output = layer->forward(input);

    // Check the output dimensions
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 1);

    // Check that the output is computed as weights * input + biases
    // This is more of an indirect verification that the output looks reasonable
    // since it's hard to compare exact values due to random weights and biases.
    EXPECT_FALSE(output.isZero()); // Just a basic check for non-zero output
}

// Test backward pass
TEST_F(DenseLayerTest, BackwardPass)
{
    // Input matrix with 3 features and 2 samples (3x2)
    Eigen::MatrixXd input(3, 1);
    input << 1, 2, 3;
    // Create a dummy gradient matrix (2x2, matching output size)
    Eigen::MatrixXd gradient(2, 1);
    gradient << 0.1, 0.2;

    // Get initial weights and biases before backward pass
    Eigen::MatrixXd initial_weights = layer->getWeights();
    Eigen::VectorXd initial_biases = layer->getBiases();

    // Perform forward pass
    Eigen::MatrixXd _ = layer->forward(input);

    // Perform backward pass
    Eigen::MatrixXd prev_layer_gradient = layer->backward(gradient, 0.01);

    // Check that the previous layer gradient has the correct dimensions (should match input size)
    ASSERT_EQ(prev_layer_gradient.rows(), 3);
    ASSERT_EQ(prev_layer_gradient.cols(), 1);

    Eigen::MatrixXd updated_weights = layer->getWeights();
    Eigen::VectorXd updated_biases = layer->getBiases();

    ASSERT_FALSE(updated_weights.isApprox(initial_weights)); // Check if weights have changed
    ASSERT_FALSE(updated_biases.isApprox(initial_biases));   // Check if biases have changed
}

// Test layer with random data
TEST_F(DenseLayerTest, RandomDataTest)
{
    // Create random input data (3x2)
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(3, 1);

    // Perform forward pass
    Eigen::MatrixXd output = layer->forward(input);

    // Ensure that the output is not zero (as it involves random weights and biases)
    EXPECT_FALSE(output.isZero());

    // Ensure that the output dimensions are correct (should be 2x2)
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 1);
}

// Test layer with zero gradients during backward pass
TEST_F(DenseLayerTest, ZeroGradientTest)
{
    // Create random input data (3x2)
    Eigen::MatrixXd input(3, 1);
    input << 1, 2, 3;

    // Create zero gradient (2x2, same size as the output)
    Eigen::MatrixXd zero_gradient(2, 1);
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

class ConvolutionalLayerTest : public ::testing::Test {
protected:
    ConvolutionalLayer* conv_layer;

    void SetUp() override {
        // Create a layer with 1 filter and kernel size 2
        conv_layer = new ConvolutionalLayer(2, 1);

        // Set deterministic kernel and bias values
        Eigen::MatrixXd kernel(2, 2);
        kernel << 1, 0, 
                   0, -1;
        std::vector<Eigen::MatrixXd> kernel_vec = { kernel };
        conv_layer->setKernels(kernel_vec);

        std::vector<double> biases = { 0.0 };
        conv_layer->setBiases(biases);
    }

    void TearDown() override {
        delete conv_layer;
    }
};

// Forward pass med same convolution
TEST_F(ConvolutionalLayerTest, ForwardSingleFilter) {
    Eigen::MatrixXd input(3, 3);
    input << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;

    // Same convolution: output = 3x3 (samma som input)
    Eigen::MatrixXd expected(3, 3);
    expected << 4, 4, -3,
                4, 4, -6,
                -7,  -8, -9;

    Eigen::MatrixXd output = conv_layer->forward(input);

    ASSERT_EQ(output.rows(), input.rows());
    ASSERT_EQ(output.cols(), input.cols());
    EXPECT_TRUE(output.isApprox(expected));
}

// Backward gradient test
TEST_F(ConvolutionalLayerTest, BackwardGradientShape) {
    Eigen::MatrixXd input(3, 3);
    input << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;

    std::vector<Eigen::MatrixXd> input_batch = { input };
    auto forward_output = conv_layer->forwardBatch(input_batch);

    // Gradient från nästa lager samma storlek som forward-output (3x3)
    Eigen::MatrixXd grad = Eigen::MatrixXd::Ones(forward_output[0].rows(), forward_output[0].cols());
    std::vector<Eigen::MatrixXd> grad_batch = { grad };

    auto input_gradient_batch = conv_layer->backwardBatch(grad_batch, 0.01);

    ASSERT_EQ(input_gradient_batch.size(), 1);
    ASSERT_EQ(input_gradient_batch[0].rows(), input.rows());
    ASSERT_EQ(input_gradient_batch[0].cols(), input.cols());
}

// Test att kernels och biases uppdateras efter backward
TEST_F(ConvolutionalLayerTest, KernelAndBiasUpdate) {
    Eigen::MatrixXd input(3, 3);
    input << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;

    std::vector<Eigen::MatrixXd> input_batch = { input };
    conv_layer->forwardBatch(input_batch);

    Eigen::MatrixXd grad = Eigen::MatrixXd::Ones(3, 3); // output = same = 3x3
    std::vector<Eigen::MatrixXd> grad_batch = { grad };

    Eigen::MatrixXd initial_kernel = conv_layer->getKernels()[0];
    double initial_bias = conv_layer->getBiases()[0];

    conv_layer->backwardBatch(grad_batch, 0.01);

    Eigen::MatrixXd updated_kernel = conv_layer->getKernels()[0];
    double updated_bias = conv_layer->getBiases()[0];

    ASSERT_FALSE(updated_kernel.isApprox(initial_kernel));
    ASSERT_NE(updated_bias, initial_bias);
}

// Test input_gradient values (kan Approx beroende på kernel)
TEST_F(ConvolutionalLayerTest, InputGradientValues) {
    Eigen::MatrixXd input(2, 2);
    input << 1, 2,
             3, 4;

    std::vector<Eigen::MatrixXd> input_batch = { input };
    conv_layer->forwardBatch(input_batch);

    // Gradient samma storlek som output (same = 2x2)
    Eigen::MatrixXd grad(2, 2);
    grad << 1, 1,
            1, 1;
    std::vector<Eigen::MatrixXd> grad_batch = { grad };

    auto input_gradient_batch = conv_layer->backwardBatch(grad_batch, 0.0); // lr=0

    ASSERT_EQ(input_gradient_batch.size(), 1);
    ASSERT_EQ(input_gradient_batch[0].rows(), 2);
    ASSERT_EQ(input_gradient_batch[0].cols(), 2);

    // Om du vill kan du beräkna expected manually med same convolution
    // Eigen::MatrixXd expected_input_gradient = ... 
    // ASSERT_TRUE(input_gradient_batch[0].isApprox(expected_input_gradient));
}

class ReshapeLayerTest : public ::testing::Test {
protected:
    ReshapeLayer *reshape_layer;

    void SetUp() override {
        // Initialize ReshapeLayer to reshape from 2x2 to 4x1
        reshape_layer = new ReshapeLayer(4, 1);
    }

    void TearDown() override {
        delete reshape_layer;
    }
};

TEST_F(ReshapeLayerTest, ForwardAndBackward) {
    Eigen::MatrixXd input(2, 2);
    input << 1, 3,
             2, 4;

    Eigen::MatrixXd expected_flattened(4, 1);
    expected_flattened << 1,
                          2,
                          3,
                          4;

    std::vector<Eigen::MatrixXd> input_batch = { input };

    // Test forward
    std::vector<Eigen::MatrixXd> output_batch = reshape_layer->forwardBatch(input_batch);
    Eigen::MatrixXd output = output_batch[0];

    ASSERT_EQ(output.rows(), 4);
    ASSERT_EQ(output.cols(), 1);
    EXPECT_TRUE(output.isApprox(expected_flattened));

    // Test backward
    Eigen::MatrixXd gradient(4, 1);
    gradient << 0.1,
                0.2,
                0.3,
                0.4;
    std::vector<Eigen::MatrixXd> gradient_batch = { gradient };
    Eigen::MatrixXd expected_input_gradient(2, 2);
    expected_input_gradient << 0.1, 0.3,
                               0.2, 0.4;

    std::vector<Eigen::MatrixXd> input_gradient_batch = reshape_layer->backwardBatch(gradient_batch, 0.01);
    Eigen::MatrixXd input_gradient = input_gradient_batch[0];
    ASSERT_EQ(input_gradient.rows(), 2);
    ASSERT_EQ(input_gradient.cols(), 2);
    EXPECT_TRUE(input_gradient.isApprox(expected_input_gradient));
}

TEST_F(ReshapeLayerTest, TestNonFlatReshape) {
    reshape_layer = new ReshapeLayer(2, 8); // Reshape back to 2x2
    Eigen::MatrixXd input(4, 4);
    input << 1, 5, 9, 13,
            2, 6, 10, 14,
            3, 7, 11, 15,
            4, 8, 12, 16;
    Eigen::MatrixXd expected_reshaped(2, 8);
    expected_reshaped << 1, 3, 5, 7, 9, 11, 13, 15,
                         2, 4, 6, 8, 10, 12, 14, 16; 

    std::vector<Eigen::MatrixXd> input_batch = { input };
    std::vector<Eigen::MatrixXd> output_batch = reshape_layer->forwardBatch(input_batch);
    Eigen::MatrixXd output = output_batch[0];
    std::cout << "Output:\n" << output << std::endl;
    ASSERT_EQ(output.rows(), 2);
    ASSERT_EQ(output.cols(), 8);
    EXPECT_TRUE(output.isApprox(expected_reshaped));
}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}