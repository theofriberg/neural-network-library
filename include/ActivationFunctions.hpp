#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include "Eigen/Dense"

namespace ActivationFunctions
{
    Eigen::MatrixXd relu(const Eigen::MatrixXd &matrix);
    Eigen::MatrixXd reluPrime(const Eigen::MatrixXd &matrix);

    Eigen::MatrixXd tanh(const Eigen::MatrixXd &matrix);
    Eigen::MatrixXd tanhPrime(const Eigen::MatrixXd &matrix);

    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &matrix);
    Eigen::MatrixXd sigmoidPrime(const Eigen::MatrixXd &matrix);

    Eigen::MatrixXd softmax(const Eigen::MatrixXd &matrix);
};

#endif // ACTIVATION_FUNCTIONS_HPP