#ifndef LAYER_HPP
#define LAYER_HPP

#include "Types.hpp"

#include <Eigen/Dense>

class Layer
{
  public:
    Layer(std::string name) : _name(name) {}
    virtual ~Layer() = default;
    virtual MatrixXdRM forward(const MatrixXdRM &input_batch) = 0;
    virtual MatrixXdRM backward(const MatrixXdRM &gradient_batch, double learning_rate = 1) = 0;

    std::string getName() { return _name; }

  private:
    std::string _name;
};

#endif // LAYER_HPP