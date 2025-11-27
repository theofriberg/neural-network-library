#ifndef RESHAPE_LAYER_HPP
#define RESHAPE_LAYER_HPP
#include <Eigen/Dense>

#include "Layer.hpp"

class ReshapeLayer : public Layer
{
  public:
    ReshapeLayer(int output_rows, int output_cols)
        : Layer("ReshapeLayer"), _output_rows(output_rows), _output_cols(output_cols)
    {
    }
    std::vector<Eigen::MatrixXd> forwardBatch(const std::vector<Eigen::MatrixXd> &input) override;
    std::vector<Eigen::MatrixXd> backwardBatch(const std::vector<Eigen::MatrixXd> &gradient_batch,
                                               double learning_rate) override;

  private:
    int _input_rows;
    int _input_cols;
    int _output_rows;
    int _output_cols;
};

#endif // RESHAPE_LAYER_HPP