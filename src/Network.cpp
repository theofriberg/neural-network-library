#include "../include/Network.hpp"

#include <fstream>
#include <iostream>

void Network::train(const std::vector<Eigen::MatrixXd> &input_batch,
                    const std::vector<Eigen::MatrixXd> &labels, int epochs, double learning_rate)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double loss = 0;
        for (int i = 0; i < input_batch.size(); i++)
        {
            Eigen::Matrix y_pred = _forward(input_batch[i]);
            loss += _loss_function(y_pred, labels[i]);
            Eigen::MatrixXd loss_gradient = _loss_function_prime(labels[i], y_pred);
            _backward(loss_gradient, learning_rate);
        }
        double error = loss / input_batch.size();
        std::cout << "Epoch: " << epoch + 1 << ", Error: " << error << std::endl;
    }
}

Eigen::MatrixXd Network::predict(const Eigen::MatrixXd &input_data) const
{
    return _forward(input_data);
}

void Network::addLayer(Layer *layer) { _layers.push_back(layer); }

int Network::getLayerCount() const { return _layers.size(); }

Eigen::MatrixXd Network::_forward(const Eigen::MatrixXd &input_data) const
{
    Eigen::MatrixXd output = input_data;
    for (auto layer : _layers)
    {
        output = layer->forward(output);
    }
    return output;
}

void Network::_backward(const Eigen::MatrixXd &gradient, double learning_rate)
{
    Eigen::MatrixXd current_gradient = gradient;
    for (auto layer = _layers.rbegin(); layer != _layers.rend(); ++layer)
    {
        current_gradient = (*layer)->backward(current_gradient, learning_rate);
    }
}
