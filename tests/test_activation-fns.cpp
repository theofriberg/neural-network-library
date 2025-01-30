#include <gtest/gtest.h>
#include "../include/ActivationFunctions.hpp"
#include "../include/Eigen/Dense"
#include <iostream>

// Test ReLU Activation Function
TEST(ActivationFunctionsTest, ReLU)
{
    Eigen::MatrixXd input(2, 2);
    input << -1, 2,
        0, -3;

    Eigen::MatrixXd expected(2, 2);
    expected << 0, 2,
        0, 0;

    Eigen::MatrixXd result = ActivationFunctions::relu(input);
    EXPECT_TRUE(result.isApprox(expected));
}

// Test ReLU Derivative
TEST(ActivationFunctionsTest, ReLU_Prime)
{
    Eigen::MatrixXd input(2, 2);
    input << -1, 2,
        0, -3;

    Eigen::MatrixXd expected(2, 2);
    expected << 0, 1,
        0, 0;

    Eigen::MatrixXd result = ActivationFunctions::reluPrime(input);
    EXPECT_TRUE(result.isApprox(expected));
}

// Test Sigmoid Activation Function
TEST(ActivationFunctionsTest, Sigmoid)
{
    Eigen::MatrixXd input(2, 2);
    input << 0, 1,
        -1, 2;

    Eigen::MatrixXd expected(2, 2);
    expected << 0.5, 1 / (1 + exp(-1)),
        1 / (1 + exp(1)), 1 / (1 + exp(-2));

    Eigen::MatrixXd result = ActivationFunctions::sigmoid(input);
    EXPECT_TRUE(result.isApprox(expected, 1e-5));
}

// Test Sigmoid Derivative
TEST(ActivationFunctionsTest, Sigmoid_Prime)
{
    Eigen::MatrixXd input(2, 2);
    input << 0, 1,
        -1, 2;

    Eigen::MatrixXd sigmoid_vals = ActivationFunctions::sigmoid(input);
    Eigen::MatrixXd expected = sigmoid_vals.array() * (1 - sigmoid_vals.array());

    Eigen::MatrixXd result = ActivationFunctions::sigmoidPrime(input);
    EXPECT_TRUE(result.isApprox(expected, 1e-5));
}

// Test Tanh Activation Function
TEST(ActivationFunctionsTest, Tanh)
{
    Eigen::MatrixXd input(2, 2);
    input << 0, 1,
        -1, 2;

    Eigen::MatrixXd expected = input.array().tanh();

    Eigen::MatrixXd result = ActivationFunctions::tanh(input);
    EXPECT_TRUE(result.isApprox(expected, 1e-5));
}

// Test Softmax Activation Function
TEST(ActivationFunctionsTest, Softmax)
{
    Eigen::MatrixXd input(2, 2);
    input << 1, 2,
        3, 4;

    Eigen::MatrixXd softmax_result = ActivationFunctions::softmax(input);
    std::cout << "Softmax result: " << softmax_result << std::endl;
    Eigen::VectorXd sum_rows = softmax_result.rowwise().sum();
    std::cout << "Sum of rows: " << sum_rows.transpose() << std::endl;

    // Each row of softmax should sum to 1
    EXPECT_TRUE(sum_rows.isApprox(Eigen::VectorXd::Ones(2), 1e-5));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}