#ifndef LOSS_FUNCTIONS_HPP
#define LOSS_FUNCTIONS_HPP

#include "Eigen/Dense"

namespace LossFunctions
{
    double mse(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred);
    Eigen::MatrixXd msePrime(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred);

    double categoricalCrossentropy(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred);
    Eigen::MatrixXd categoricalCrossentropyPrime(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred);
};

#endif // LOSS_FUNCTIONS_HPP