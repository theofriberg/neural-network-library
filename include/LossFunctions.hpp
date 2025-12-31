#ifndef LOSS_FUNCTIONS_HPP
#define LOSS_FUNCTIONS_HPP

#include "Types.hpp"
#include <Eigen/Dense>

namespace LossFunctions
{
double mse(const MatrixXdRM &y_true, const MatrixXdRM &y_pred);
MatrixXdRM msePrime(const MatrixXdRM &y_true, const MatrixXdRM &y_pred);

double categoricalCrossentropy(const MatrixXdRM &y_true, const MatrixXdRM &y_pred);
MatrixXdRM categoricalCrossentropyPrime(const MatrixXdRM &y_true, const MatrixXdRM &y_pred);
}; // namespace LossFunctions

#endif // LOSS_FUNCTIONS_HPP