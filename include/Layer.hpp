#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>

class Layer {
public:
  Layer(std::string name) : _name(name) {}
  virtual ~Layer() = default;
  virtual std::vector<Eigen::MatrixXd> forwardBatch(const std::vector<Eigen::MatrixXd> &input_batch) = 0;
  virtual std::vector<Eigen::MatrixXd> backwardBatch(const std::vector<Eigen::MatrixXd> &gradient_batch,
                                   double learning_rate = 1) = 0;
  Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
    return forwardBatch(std::vector<Eigen::MatrixXd>{input})[0];
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &gradient,
                           double learning_rate = 1) {
    return backwardBatch(std::vector<Eigen::MatrixXd>{gradient}, learning_rate)[0];
  }

  std::string getName() { return _name; }

private:
  std::string _name;
};

#endif // LAYER_HPP