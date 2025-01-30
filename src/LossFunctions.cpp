#include "../include/Eigen/Dense"
#include "../include/LossFunctions.hpp"

#include <iostream>

double LossFunctions::mse(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred)
{
    Eigen::MatrixXd error_mat = y_true - y_pred;
    Eigen::MatrixXd error_squared = error_mat.array().square();
    return error_squared.mean();
}

Eigen::MatrixXd LossFunctions::msePrime(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred)
{
    Eigen::MatrixXd error_mat = y_pred - y_true; // Notice this other way around compared to above, because inner derivative gives a minus sign
    double n = error_mat.size();
    return (2.0 / n) * error_mat;
}

double LossFunctions::categoricalCrossentropy(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred)
{
    Eigen::MatrixXd y_pred_clamped = y_pred.cwiseMax(1e-7).cwiseMin(1 - 1e-7);
    Eigen::MatrixXd loss = -(y_true.array() * y_pred_clamped.array().log());
    return loss.mean();
}

Eigen::MatrixXd LossFunctions::categoricalCrossentropyPrime(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred)
{
    Eigen::MatrixXd y_pred_clamped = y_pred.cwiseMax(1e-7).cwiseMin(1 - 1e-7);
    return -(y_true.array() / y_pred_clamped.array());
}