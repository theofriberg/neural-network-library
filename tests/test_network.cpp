#include "DenseLayer.hpp"
#include "Eigen/Dense"
#include "LossFunctions.hpp"
#include "Network.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(NetworkTest, AddLayer)
{
    Network net(LossFunctions::mse, LossFunctions::msePrime);
    DenseLayer layer(2, 3);
    net.addLayer(&layer);
    EXPECT_EQ(net.getLayerCount(), 1);
}

TEST(NetworkTest, Predict)
{
    Network net(LossFunctions::mse, LossFunctions::msePrime);

    DenseLayer layer(2, 3);
    net.addLayer(&layer);

    Eigen::MatrixXd input(2, 1);
    input << 1, 2;

    Eigen::MatrixXd output = net.predict(input);
    ASSERT_EQ(output.rows(), 3);
    ASSERT_EQ(output.cols(), 1);
}

TEST(NetworkTest, TrainRunsWithoutErrors)
{
    Network net(LossFunctions::mse, LossFunctions::msePrime);

    DenseLayer layer(2, 3);
    net.addLayer(&layer);

    std::vector<Eigen::MatrixXd> inputs = {Eigen::MatrixXd::Random(2, 1)};
    std::vector<Eigen::MatrixXd> labels = {Eigen::MatrixXd::Random(3, 1)};

    EXPECT_NO_THROW(net.train(inputs, labels, 100, 0.01));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
