#include "../include/LossFunctions.hpp"
#include <Eigen/Dense>

#include <iostream>

double LossFunctions::mse(const MatrixXdRM &y_true, const MatrixXdRM &y_pred)
{
    auto error_mat = y_true - y_pred;
    auto error_squared = error_mat.array().square();
    return error_squared.mean();
}

MatrixXdRM LossFunctions::msePrime(const MatrixXdRM &y_true, const MatrixXdRM &y_pred)
{
    auto error_mat = y_pred - y_true; // Notice this other way around compared to above,
                                      // because inner derivative gives a minus sign
    double n = error_mat.size();
    return (2.0 / n) * error_mat;
}

double LossFunctions::categoricalCrossentropy(const MatrixXdRM &y_true, const MatrixXdRM &y_pred)
{
    auto y_pred_clamped = y_pred.cwiseMax(1e-7).cwiseMin(1 - 1e-7);
    auto loss = -(y_true.array() * y_pred_clamped.array().log());
    return loss.mean();
}

MatrixXdRM LossFunctions::categoricalCrossentropyPrime(const MatrixXdRM &y_true,
                                                       const MatrixXdRM &y_pred)
{
    auto y_pred_clamped = y_pred.cwiseMax(1e-7).cwiseMin(1 - 1e-7);
    return -(y_true.array() / y_pred_clamped.array());
}