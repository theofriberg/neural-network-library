#include "../include/Network.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

void Network::train(const std::vector<Eigen::MatrixXd> &input_batch,
                    const std::vector<Eigen::MatrixXd> &labels, int epochs, double learning_rate)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::vector<Eigen::MatrixXd> y_pred_batch = _forward(input_batch);
        std::vector<Eigen::MatrixXd> gradient_batch(y_pred_batch.size());
        for (int i = 0; i < y_pred_batch.size(); i++)
        {
            Eigen::MatrixXd loss_gradient = _loss_function_prime(labels[i], y_pred_batch[i]);
            gradient_batch[i] = loss_gradient;
            double loss = _loss_function(labels[i], y_pred_batch[i]);
            std::cout << "Loss = " << loss << std::endl;
        }
        _backward(gradient_batch, learning_rate);
    }
}

Eigen::MatrixXd Network::predict(const Eigen::MatrixXd &input_data) const
{
    std::vector<Eigen::MatrixXd> input_wrap = {input_data};
    return _forward(input_wrap)[0];
}

void Network::saveModel(std::string &filename) const
{
    throw std::logic_error("saveModel() not yet implemented");
}

void Network::loadModel(std::string &filename)
{
    throw std::logic_error("loadModel() not yet implemented");
}

void Network::addLayer(Layer *layer) { _layers.push_back(layer); }

int Network::getLayerCount() const { return _layers.size(); }

std::vector<Eigen::MatrixXd>
Network::_forward(const std::vector<Eigen::MatrixXd> &input_batch) const
{
    std::vector<Eigen::MatrixXd> output(input_batch.size());
    for (auto layer : _layers)
    {
        output = layer->forwardBatch(input_batch);
    }
    return output;
}

void Network::_backward(const std::vector<Eigen::MatrixXd> &gradient_batch, double learning_rate)
{
    std::vector<Eigen::MatrixXd> current_gradient = gradient_batch;
    for (auto layer = _layers.rbegin(); layer != _layers.rend(); ++layer)
    {
        current_gradient = (*layer)->backwardBatch(current_gradient, learning_rate);
    }
}
