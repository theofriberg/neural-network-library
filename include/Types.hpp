#ifndef TYPES_HPP
#define TYPES_HPP

#include <Eigen/Dense>

using MatrixXdRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;
using MatrixXdRMValuedFunc = std::function<MatrixXdRM(const MatrixXdRM &)>;
using LossFunction = std::function<double(const MatrixXdRM &, const MatrixXdRM &)>;
using LossFunctionPrime = std::function<MatrixXdRM(const MatrixXdRM &, const MatrixXdRM &)>;

#endif // TYPES_HPP