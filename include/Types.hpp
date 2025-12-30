#ifndef TYPES_HPP
#define TYPES_HPP

#include <Eigen/Dense>

using MatrixXdRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXdRM = Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>;

#endif // TYPES_HPP