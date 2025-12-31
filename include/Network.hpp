#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "Layer.hpp"
#include "Types.hpp"
#include <Eigen/Dense>
#include <vector>

using LayerPtr = std::shared_ptr<Layer>;

class Network
{
  public:
    Network(LossFunction loss_fn, LossFunctionPrime loss_fn_prime)
        : _loss_function(loss_fn), _loss_function_prime(loss_fn_prime)
    {
    }

    void train(const MatrixXdRM &input_batch, const MatrixXdRM &labels, int epochs,
               double learning_rate);
    MatrixXdRM predict(const MatrixXdRM &input_batch) const;
    void addLayer(LayerPtr layer);
    void saveModel(std::string &filename) const;
    void loadModel(std::string &filename);
    int getLayerCount() const;

  private:
    std::vector<LayerPtr> _layers;

    // Loss function and its derivative
    LossFunction _loss_function;
    LossFunctionPrime _loss_function_prime;

    MatrixXdRM _forward(const MatrixXdRM &input_batch) const;
    void _backward(const MatrixXdRM &gradient_batch, double learning_rate);
};

#endif // NETWORK_HPP