#include <gtest/gtest.h>
#define UNIT_TEST

#include "../include/ActivationLayers.hpp"
#include "../include/ConvolutionalLayer.hpp"
#include "../include/DenseLayer.hpp"
#include <Eigen/Dense>
#include <iostream>

class DenseLayerTest : public ::testing::Test
{
  protected:
    // Set up a DenseLayer with 3 inputs and 2 outputs (arbitrary values for testing)
    DenseLayer *layer;

    void SetUp() override { layer = new DenseLayer(3, 2); }

    void TearDown() override { delete layer; }
};

// Test forward pass
TEST_F(DenseLayerTest, ForwardPass)
{
    // Input med 3 features och 2 samples (3x2)
    MatrixXdRM input(3, 2);
    input << 1, 4, 2, 5, 3, 6;

    MatrixXdRM output = layer->forward(input);

    // Output ska ha 2 rader (output features) och 2 kolumner (batch)
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 2);
    EXPECT_FALSE(output.isZero());
}

// Test backward pass
TEST_F(DenseLayerTest, BackwardPass)
{
    // Input med 3 features och 2 samples (3x2)
    MatrixXdRM input(3, 2);
    input << 1, 4, 2, 5, 3, 6;

    // Dummy gradient (output features x batch)
    MatrixXdRM gradient(2, 2);
    gradient << 0.1, 0.2, 0.3, 0.4;

    MatrixXdRM initial_weights = layer->getWeights();
    VectorXd initial_biases = layer->getBiases();

    // Forward pass
    MatrixXdRM _ = layer->forward(input);

    // Backward pass
    MatrixXdRM prev_layer_grad = layer->backward(gradient, 0.01);

    // Kontrollera att dimensions matchar input
    ASSERT_EQ(prev_layer_grad.rows(), 3);
    ASSERT_EQ(prev_layer_grad.cols(), 2);

    // Kontrollera att vikter och bias uppdaterats
    MatrixXdRM updated_weights = layer->getWeights();
    VectorXd updated_biases = layer->getBiases();
    ASSERT_FALSE(updated_weights.isApprox(initial_weights));
    ASSERT_FALSE(updated_biases.isApprox(initial_biases));
}

// Test layer med random batch
TEST_F(DenseLayerTest, RandomBatchTest)
{
    MatrixXdRM input = MatrixXdRM::Random(3, 5); // 5 samples i batch
    MatrixXdRM output = layer->forward(input);

    EXPECT_FALSE(output.isZero());
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 5);
}

// Test zero gradient
TEST_F(DenseLayerTest, ZeroGradientTest)
{
    MatrixXdRM input(3, 2);
    input << 1, 4, 2, 5, 3, 6;

    MatrixXdRM zero_grad = MatrixXdRM::Zero(2, 2);

    MatrixXdRM initial_weights = layer->getWeights();
    VectorXd initial_biases = layer->getBiases();

    layer->forward(input);
    MatrixXdRM prev_layer_grad = layer->backward(zero_grad, 0.01);

    ASSERT_TRUE(prev_layer_grad.isZero());
    EXPECT_TRUE(initial_weights.isApprox(layer->getWeights()));
    EXPECT_TRUE(initial_biases.isApprox(layer->getBiases()));
}

class ActivationLayerTest : public ::testing::Test
{
  protected:
    // Set up a ReLU activation layer
    ReLU *relu_layer;

    void SetUp() override { relu_layer = new ReLU(); }

    void TearDown() override { delete relu_layer; }
};

TEST_F(ActivationLayerTest, ReLUForward)
{
    MatrixXdRM input(2, 2);
    input << -1, 2, 0, -3;

    MatrixXdRM expected(2, 2);
    expected << 0, 2, 0, 0;

    // Ensure forward gives expected output
    MatrixXdRM output = relu_layer->forward(input);
    EXPECT_TRUE(output.isApprox(expected));
}

TEST_F(ActivationLayerTest, ReLUBackward)
{
    MatrixXdRM input(2, 2);
    input << -1, 2, 0, -3;

    MatrixXdRM gradient(2, 2);
    gradient << 0.1, 0.2, 0.3, 0.4;

    MatrixXdRM expected_gradient(2, 2);
    expected_gradient << 0, 0.2, 0, 0;

    // Perform a forward pass
    MatrixXdRM _ = relu_layer->forward(input);

    // Ensure backward gives expected gradient
    MatrixXdRM relu_gradient = relu_layer->backward(gradient);
    EXPECT_TRUE(relu_gradient.isApprox(expected_gradient));
}

class ConvolutionalLayerTest : public ::testing::Test
{
  protected:
    ConvolutionalLayer *conv_layer;

    void SetUp() override
    {
        // kernel_size = 2, num_filters = 2, input = 3x3
        conv_layer = new ConvolutionalLayer(2, 2, 3, 3);

#ifdef UNIT_TEST
        MatrixXdRM kernel1(2, 2);
        kernel1 << 1, 0, 0, -1;

        MatrixXdRM kernel2(2, 2);
        kernel2 << 0, 1, -1, 0;

        std::vector<MatrixXdRM> kernels = {kernel1, kernel2};
        conv_layer->setKernels(kernels);

        std::vector<double> biases = {0.0, 0.0};
        conv_layer->setBiases(biases);
#endif
    }

    void TearDown() override { delete conv_layer; }
};

TEST_F(ConvolutionalLayerTest, ForwardOutputShape)
{
    // Input: 3x3 → flattened → (9 x 1)
    MatrixXdRM input(9, 1);
    input << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    MatrixXdRM output = conv_layer->forward(input);

    // Output: num_filters * H * W = 2 * 3 * 3 = 18
    ASSERT_EQ(output.rows(), 18);
    ASSERT_EQ(output.cols(), 1);
}

TEST_F(ConvolutionalLayerTest, BackwardInputGradientShape)
{
    MatrixXdRM input(9, 1);
    input << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    MatrixXdRM output = conv_layer->forward(input);

    // Gradient från nästa lager: samma form som output
    MatrixXdRM grad = MatrixXdRM::Ones(output.rows(), output.cols());

    MatrixXdRM input_grad = conv_layer->backward(grad, 0.01);

    // Input-gradient ska matcha input-shape
    ASSERT_EQ(input_grad.rows(), 9);
    ASSERT_EQ(input_grad.cols(), 1);
}

TEST_F(ConvolutionalLayerTest, KernelAndBiasAreUpdated)
{
    MatrixXdRM input(9, 1);
    input << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    conv_layer->forward(input);

    MatrixXdRM grad = MatrixXdRM::Ones(18, 1); // 2 filters * 3 * 3

#ifdef UNIT_TEST
    MatrixXdRM kernel_before = conv_layer->getKernels()[0];
    double bias_before = conv_layer->getBiases()[0];
#endif

    conv_layer->backward(grad, 0.01);

#ifdef UNIT_TEST
    MatrixXdRM kernel_after = conv_layer->getKernels()[0];
    double bias_after = conv_layer->getBiases()[0];

    ASSERT_FALSE(kernel_after.isApprox(kernel_before));
    ASSERT_NE(bias_after, bias_before);
#endif
}

TEST_F(ConvolutionalLayerTest, InputGradientFiniteValues)
{
    MatrixXdRM input(4, 1); // 2x2
    input << 1, 2, 3, 4;

    // Nytt lager med 2x2 input
    delete conv_layer;
    conv_layer = new ConvolutionalLayer(2, 2, 2, 2);

    MatrixXdRM output = conv_layer->forward(input);
    MatrixXdRM grad = MatrixXdRM::Ones(output.rows(), 1);

    MatrixXdRM input_grad = conv_layer->backward(grad, 0.0);

    ASSERT_EQ(input_grad.rows(), 4);
    ASSERT_EQ(input_grad.cols(), 1);
    ASSERT_TRUE(input_grad.allFinite());
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}