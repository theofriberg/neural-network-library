#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/CorrelationCalculator.hpp"

TEST(CorrelationCalculatorTest, CorrelationSame) {
    Eigen::MatrixXd subject(3, 3);
    subject << 1, 2, 3,
               4, 5, 6,
               7, 8, 9;

    Eigen::MatrixXd kernel(2, 2);
    kernel << 0, 1,
              2, 3;

    Eigen::MatrixXd expected(3, 3);
    expected << 25, 31, 12,
                43, 49, 18,
                8,  9,  0;


    Eigen::MatrixXd result = CorrelationCalculator::crossCorrelate(subject, kernel);
    EXPECT_TRUE(result.isApprox(expected));
}

TEST(CorrelationCalculatorTest, CorrelationValid) {
    Eigen::MatrixXd subject(3, 3);
    subject << 1, 2, 3,
               4, 5, 6,
               7, 8, 9;

    Eigen::MatrixXd kernel(2, 2);
    kernel << 0, 1,
              2, 3;

    Eigen::MatrixXd expected(2, 2);
    expected << 25, 31,
                43, 49;

    Eigen::MatrixXd result = CorrelationCalculator::crossCorrelate(subject, kernel, 2, 2);
    EXPECT_TRUE(result.isApprox(expected));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}