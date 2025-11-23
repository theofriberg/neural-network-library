#include "../include/CorrelationCalculator.hpp"
#include <iostream>

Eigen::MatrixXd CorrelationCalculator::convolve(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel) {
    Eigen::MatrixXd kernel_rot = rotate180(kernel);
    return crossCorrelate(subject, kernel_rot);
}

Eigen::MatrixXd CorrelationCalculator::crossCorrelate(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel) {
    int kernel_rows = kernel.rows();
    int kernel_cols = kernel.cols();
    int pad_rows = kernel_rows - 1;
    int pad_cols = kernel_cols - 1;
    int pad_top = pad_rows / 2;
    int pad_bottom = pad_rows - pad_top;
    int pad_left = pad_cols / 2;
    int pad_right = pad_cols - pad_left;

    Eigen::MatrixXd padded_subject = Eigen::MatrixXd::Zero(subject.rows() + pad_rows, subject.cols() + pad_cols);
    padded_subject.block(pad_top, pad_left, subject.rows(), subject.cols()) = subject;

    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(subject.rows(), subject.cols());
    Eigen::VectorXd kernel_vector = kernel.reshaped();
    for (int i = 0; i < subject.rows(); i++)
    {
        for (int j = 0; j < subject.cols(); j++)
        {
            Eigen::MatrixXd region = padded_subject.block(i, j, kernel_rows, kernel_cols);
            Eigen::VectorXd region_vector = region.reshaped();
            output(i, j) = region_vector.dot(kernel_vector);
        }
        
    }
    
    return output;
}

Eigen::MatrixXd CorrelationCalculator::rotate180(const Eigen::MatrixXd &matrix) {
    return matrix.colwise().reverse().rowwise().reverse();
}