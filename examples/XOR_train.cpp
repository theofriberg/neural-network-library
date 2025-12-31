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
    MatrixXdRM input(2, 4);
    input << 1, 0, 1, 0, 0, 1, 1, 0;

    MatrixXdRM labels(1, 4);
    labels << 1, 1, 0, 0;

    Network model(LossFunctions::mse, LossFunctions::msePrime);

    model.addLayer(std::make_shared<DenseLayer>(2, 3));
    model.addLayer(std::make_shared<Tanh>());
    model.addLayer(std::make_shared<DenseLayer>(3, 1));
    model.addLayer(std::make_shared<Tanh>());

    model.train(input, labels, 1000, 0.5);

    std::cout << "Predictions:" << std::endl;
    std::cout << "1, 0: " << std::abs(std::round(model.predict(input)(0, 0))) << std::endl;
    std::cout << "0, 1: " << std::abs(std::round(model.predict(input)(0, 1))) << std::endl;
    std::cout << "1, 1: " << std::abs(std::round(model.predict(input)(0, 2))) << std::endl;
    std::cout << "0, 0: " << std::abs(std::round(model.predict(input)(0, 3))) << std::endl;
}