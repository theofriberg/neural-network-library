#include "ActivationFunctions.hpp"
#include <Eigen/Dense>

MatrixXdRM ActivationFunctions::relu(const MatrixXdRM &matrix) { return matrix.cwiseMax(0); }

MatrixXdRM ActivationFunctions::reluPrime(const MatrixXdRM &matrix)
{
    return (matrix.array() > 0).cast<double>();
}

MatrixXdRM ActivationFunctions::tanh(const MatrixXdRM &matrix)
{
    auto mat_exp = (2 * matrix.array()).exp();
    return (mat_exp.array() - 1) / (mat_exp.array() + 1);
}

MatrixXdRM ActivationFunctions::tanhPrime(const MatrixXdRM &matrix)
{
    auto tanh_square = tanh(matrix).array().square();
    return 1 - tanh_square.array();
}

MatrixXdRM ActivationFunctions::sigmoid(const MatrixXdRM &matrix)
{
    auto mat_exp = (-matrix.array()).exp();
    return 1 / (1 + mat_exp.array());
}

MatrixXdRM ActivationFunctions::sigmoidPrime(const MatrixXdRM &matrix)
{
    MatrixXdRM s = sigmoid(matrix);
    return s.array() * (1.0 - s.array());
}

MatrixXdRM ActivationFunctions::softmax(const MatrixXdRM &matrix)
{
    VectorXd col_max_vals = matrix.colwise().maxCoeff();
    MatrixXdRM stabilized = matrix.array().rowwise() - col_max_vals.transpose().array();
    MatrixXdRM exp_vals = stabilized.array().exp();
    VectorXd col_sums = exp_vals.colwise().sum();
    return exp_vals.array().rowwise() / col_sums.transpose().array();
}
