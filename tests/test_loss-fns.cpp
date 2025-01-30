#include <gtest/gtest.h>
#include "../include/Eigen/Dense"
#include "../include/LossFunctions.hpp"

TEST(LossFunctionsTest, MSE)
{
    Eigen::MatrixXd y_true(2, 2);
    y_true << 1, 2,
        3, 4;

    Eigen::MatrixXd y_pred(2, 2);
    y_pred << 0.5, 1.5,
        2.5, 3.5;

    EXPECT_NEAR(LossFunctions::mse(y_true, y_pred), 0.25, 1e-6);
}

TEST(LossFunctionTest, MSEPrime)
{
    Eigen::MatrixXd y_true(2, 2);
    y_true << 1, 2,
        3, 4;

    Eigen::MatrixXd y_pred(2, 2);
    y_pred << 0.5, 1.5,
        2.5, 3.5;

    Eigen::MatrixXd expected(2, 2);
    expected << -0.25, -0.25,
        -0.25, -0.25;

    EXPECT_TRUE(LossFunctions::msePrime(y_true, y_pred).isApprox(expected, 1e-6));
}

TEST(LossFunctionsTest, CategoricalCrossEntropy)
{
    Eigen::MatrixXd y_true(2, 2);
    y_true << 0, 1,
        1, 0;

    Eigen::MatrixXd y_pred(2, 2);
    y_pred << 0.5, 0.5,
        0.5, 0.5;

    EXPECT_NEAR(LossFunctions::categoricalCrossentropy(y_true, y_pred), 0.34657359, 1e-6);
}

TEST(LossFunctionsTest, CategoricalCrossEntropyPrime)
{
    Eigen::MatrixXd y_true(2, 2);
    y_true << 0, 1,
        1, 0;

    Eigen::MatrixXd y_pred(2, 2);
    y_pred << 0.5, 0.5,
        0.5, 0.5;

    Eigen::MatrixXd expected(2, 2);
    expected << 0, -2,
        -2, 0;

    EXPECT_TRUE(LossFunctions::categoricalCrossentropyPrime(y_true, y_pred).isApprox(expected, 1e-6));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}