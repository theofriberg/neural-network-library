#include "../include/Eigen/Dense"
#include "../include/ActivationFunctions.hpp"

Eigen::MatrixXd ActivationFunctions::relu(const Eigen::MatrixXd &matrix)
{
    return matrix.cwiseMax(0);
}

Eigen::MatrixXd ActivationFunctions::relu_prime(const Eigen::MatrixXd &matrix)
{
    return (matrix.array() > 0).cast<double>();
}

Eigen::MatrixXd ActivationFunctions::tanh(const Eigen::MatrixXd &matrix)
{
    Eigen::MatrixXd mat_exp = (2 * matrix.array()).exp();
    return (mat_exp.array() - 1) / (mat_exp.array() + 1);
}

Eigen::MatrixXd ActivationFunctions::tanh_prime(const Eigen::MatrixXd &matrix)
{
    Eigen::MatrixXd tanh_square = tanh(matrix).array().square();
    return 1 - tanh_square.array();
}

Eigen::MatrixXd ActivationFunctions::sigmoid(const Eigen::MatrixXd &matrix)
{
    Eigen::MatrixXd mat_exp = (-matrix.array()).exp();
    return 1 / (1 + mat_exp.array());
}

Eigen::MatrixXd ActivationFunctions::sigmoid_prime(const Eigen::MatrixXd &matrix)
{
    Eigen::MatrixXd sig_vals = sigmoid(matrix).array();
    return sig_vals.array() * (1 - sig_vals.array());
}

Eigen::MatrixXd ActivationFunctions::softmax(const Eigen::MatrixXd &matrix)
{
    Eigen::VectorXd row_max_vals = matrix.rowwise().maxCoeff();
    Eigen::MatrixXd stabilized_matrix = matrix.rowwise() - row_max_vals.transpose();
    Eigen::MatrixXd softmax_values = stabilized_matrix.array().exp();
    Eigen::VectorXd sum_softmax_values = softmax_values.rowwise().sum();
    return softmax_values.array().rowwise() / sum_softmax_values.transpose().array();
}