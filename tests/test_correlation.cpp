#include <gtest/gtest.h>
#include "../include/CorrelationCalculator.hpp"
#include "../include/Eigen/Dense"

TEST(CorrelationCalculatorTest, CorrelationValid) {
    Eigen::MatrixXd subject(3, 3);
    subject << 1, 2, 3,
               4, 5, 6,
               7, 8, 9;

    Eigen::MatrixXd kernel(2, 2);
    kernel << 0, 1,
              2, 3;

    Eigen::MatrixXd expected(4, 4);
    expected << 3, 8, 13, 6,
                13, 25, 31, 12,
                25, 43, 49, 18,
                7, 8, 9, 0;

    Eigen::MatrixXd result = CorrelationCalculator::crossCorrelate(subject, kernel);
    EXPECT_TRUE(result.isApprox(expected));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}