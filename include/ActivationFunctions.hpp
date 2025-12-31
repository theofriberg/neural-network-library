#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include "Types.hpp"

#include <Eigen/Dense>

namespace ActivationFunctions
{
MatrixXdRM relu(const MatrixXdRM &matrix);
MatrixXdRM reluPrime(const MatrixXdRM &matrix);

MatrixXdRM tanh(const MatrixXdRM &matrix);
MatrixXdRM tanhPrime(const MatrixXdRM &matrix);

MatrixXdRM sigmoid(const MatrixXdRM &matrix);
MatrixXdRM sigmoidPrime(const MatrixXdRM &matrix);

MatrixXdRM softmax(const MatrixXdRM &matrix);
}; // namespace ActivationFunctions

#endif // ACTIVATION_FUNCTIONS_HPP