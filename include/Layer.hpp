#ifndef LAYER_HPP
#define LAYER_HPP

#include "Eigen/Dense"

class Layer {
public:
  Layer(std::string name) : _name(name) {}
  virtual ~Layer() = default;
  virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input) = 0;
  virtual Eigen::MatrixXd backward(const Eigen::MatrixXd &gradient,
                                   double learning_rate = 1) = 0;

  std::string getName() { return _name; }

private:
  std::string _name;
};

#endif // LAYER_HPP