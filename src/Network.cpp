#include "../include/Network.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

void Network::train(const MatrixXdRM &input_batch, const MatrixXdRM &labels, int epochs,
                    double learning_rate)
{
    std::cout << "Börjar träna" << std::endl;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        auto predictions = _forward(input_batch);
        auto loss_gradient = _loss_function_prime(predictions, labels);
        _backward(loss_gradient, learning_rate);
        if (epoch % 10 == 0)
        {
            auto loss = _loss_function(predictions, labels);
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }
}

MatrixXdRM Network::predict(const MatrixXdRM &input_batch) const { return _forward(input_batch); }

void Network::saveModel(std::string &filename) const
{
    throw std::logic_error("saveModel() not yet implemented");
}

void Network::loadModel(std::string &filename)
{
    throw std::logic_error("loadModel() not yet implemented");
}

void Network::addLayer(LayerPtr layer) { _layers.push_back(layer); }

int Network::getLayerCount() const { return _layers.size(); }

MatrixXdRM Network::_forward(const MatrixXdRM &input_batch) const
{
    MatrixXdRM output = input_batch;
    for (auto &layer : _layers)
    {
        output = layer->forward(output);
    }
    return output;
}

void Network::_backward(const MatrixXdRM &gradient_batch, double learning_rate)
{
    MatrixXdRM current_gradient = gradient_batch;
    for (auto layer = _layers.rbegin(); layer != _layers.rend(); ++layer)
    {
        current_gradient = (*layer)->backward(current_gradient, learning_rate);
    }
}
