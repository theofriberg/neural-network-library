#include "ActivationLayer.hpp"
#include "ActivationLayers.hpp"
#include "DenseLayer.hpp"
#include "Eigen/Dense"
#include "LossFunctions.hpp"
#include "Network.hpp"
#include <cmath>
#include <iostream>
#include <valarray>
#include <vector>

int main()
{
    std::vector<Eigen::MatrixXd> input = {Eigen::MatrixXd(2, 1), Eigen::MatrixXd(2, 1),
                                          Eigen::MatrixXd(2, 1), Eigen::MatrixXd(2, 1)};

    std::vector<Eigen::MatrixXd> labels = {Eigen::MatrixXd(1, 1), Eigen::MatrixXd(1, 1),
                                           Eigen::MatrixXd(1, 1), Eigen::MatrixXd(1, 1)};

    input[0] << 1, 0;
    input[1] << 0, 1;
    input[2] << 1, 1;
    input[3] << 0, 0;

    labels[0] << 1;
    labels[1] << 1;
    labels[2] << 0;
    labels[3] << 0;

    Network model(LossFunctions::mse, LossFunctions::msePrime);

    model.addLayer(new DenseLayer(2, 3));
    model.addLayer(new Tanh());
    model.addLayer(new DenseLayer(3, 1));
    model.addLayer(new Tanh());

    model.train(input, labels, 1000, 0.5);

    std::cout << "Predictions:" << std::endl;
    std::cout << "1, 0: " << std::abs(std::round(model.predict(input[0])(0, 0))) << std::endl;
    std::cout << "0, 1: " << std::abs(std::round(model.predict(input[1])(0, 0))) << std::endl;
    std::cout << "1, 1: " << std::abs(std::round(model.predict(input[2])(0, 0))) << std::endl;
    std::cout << "0, 0: " << std::abs(std::round(model.predict(input[3])(0, 0))) << std::endl;
}