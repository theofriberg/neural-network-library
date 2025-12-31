#include "DenseLayer.hpp"
#include "LossFunctions.hpp"
#include "Network.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <vector>

TEST(NetworkTest, AddLayer)
{
    Network net(LossFunctions::mse, LossFunctions::msePrime);
    net.addLayer(std::make_shared<DenseLayer>(2, 3));
    EXPECT_EQ(net.getLayerCount(), 1);
}

TEST(NetworkTest, Predict)
{
    Network net(LossFunctions::mse, LossFunctions::msePrime);
    net.addLayer(std::make_shared<DenseLayer>(2, 2));

    MatrixXdRM input(2, 2);
    input << 1, 2, 3, 4;

    MatrixXdRM output = net.predict(input);
    ASSERT_EQ(output.rows(), 2);
    ASSERT_EQ(output.cols(), 2);
}

TEST(NetworkTest, TrainRunsWithoutErrors)
{
    Network net(LossFunctions::mse, LossFunctions::msePrime);
    net.addLayer(std::make_shared<DenseLayer>(2, 1));

    MatrixXdRM inputs = MatrixXdRM::Random(2, 1);
    MatrixXdRM labels = MatrixXdRM::Random(1, 1);

    EXPECT_NO_THROW(net.train(inputs, labels, 100, 0.01));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
