#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>

class Network
{
  public:
    Network(std::function<double(Eigen::MatrixXd, Eigen::MatrixXd)> loss_fn,
            std::function<Eigen::MatrixXd(Eigen::MatrixXd, Eigen::MatrixXd)> loss_fn_prime)
        : _loss_function(loss_fn), _loss_function_prime(loss_fn_prime)
    {
    }

    void train(const std::vector<Eigen::MatrixXd> &input_batch,
               const std::vector<Eigen::MatrixXd> &labels, int epochs, double learning_rate);
    Eigen::MatrixXd predict(const Eigen::MatrixXd &input_data) const;
    void addLayer(Layer *layer);
    void saveModel(std::string &filename) const;
    void loadModel(std::string &filename);
    int getLayerCount() const;

  private:
    std::vector<Layer *> _layers;

    // Loss function and its derivative
    std::function<double(Eigen::MatrixXd, Eigen::MatrixXd)> _loss_function;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd, Eigen::MatrixXd)> _loss_function_prime;

    Eigen::MatrixXd _forward(const Eigen::MatrixXd &input_data) const;
    void _backward(const Eigen::MatrixXd &gradient, double learning_rate);
};

#endif // NETWORK_HPP