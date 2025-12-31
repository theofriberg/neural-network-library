#include "../include/ActivationFunctions.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

// Test ReLU Activation Function
TEST(ActivationFunctionsTest, ReLU)
{
    MatrixXdRM input(2, 2);
    input << -1, 2, 0, -3;

    MatrixXdRM expected(2, 2);
    expected << 0, 2, 0, 0;

    MatrixXdRM result = ActivationFunctions::relu(input);
    EXPECT_TRUE(result.isApprox(expected));
}

// Test ReLU Derivative
TEST(ActivationFunctionsTest, ReLU_Prime)
{
    MatrixXdRM input(2, 2);
    input << -1, 2, 0, -3;

    MatrixXdRM expected(2, 2);
    expected << 0, 1, 0, 0;

    MatrixXdRM result = ActivationFunctions::reluPrime(input);
    EXPECT_TRUE(result.isApprox(expected));
}

// Test Sigmoid Activation Function
TEST(ActivationFunctionsTest, Sigmoid)
{
    MatrixXdRM input(2, 2);
    input << 0, 1, -1, 2;

    MatrixXdRM expected(2, 2);
    expected << 0.5, 1 / (1 + exp(-1)), 1 / (1 + exp(1)), 1 / (1 + exp(-2));

    MatrixXdRM result = ActivationFunctions::sigmoid(input);
    EXPECT_TRUE(result.isApprox(expected, 1e-5));
}

// Test Sigmoid Derivative
TEST(ActivationFunctionsTest, Sigmoid_Prime)
{
    MatrixXdRM input(2, 2);
    input << 0, 1, -1, 2;

    MatrixXdRM sigmoid_vals = ActivationFunctions::sigmoid(input);
    MatrixXdRM expected = sigmoid_vals.array() * (1 - sigmoid_vals.array());

    MatrixXdRM result = ActivationFunctions::sigmoidPrime(input);
    EXPECT_TRUE(result.isApprox(expected, 1e-5));
}

// Test Tanh Activation Function
TEST(ActivationFunctionsTest, Tanh)
{
    MatrixXdRM input(2, 2);
    input << 0, 1, -1, 2;

    MatrixXdRM expected = input.array().tanh();

    MatrixXdRM result = ActivationFunctions::tanh(input);
    EXPECT_TRUE(result.isApprox(expected, 1e-5));
}

// Test Softmax Activation Function
TEST(ActivationFunctionsTest, Softmax)
{
    MatrixXdRM input(2, 2);
    input << 1, 2, 3, 4;

    MatrixXdRM softmax_result = ActivationFunctions::softmax(input);
    std::cout << "Softmax result: " << softmax_result << std::endl;
    Eigen::VectorXd sum_cols = softmax_result.colwise().sum();
    std::cout << "Sum of columns: " << sum_cols.transpose() << std::endl;

    // Each column of softmax should sum to 1
    EXPECT_TRUE(sum_cols.isApprox(Eigen::VectorXd::Ones(2), 1e-5));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}