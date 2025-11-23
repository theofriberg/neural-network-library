#include "../include/CorrelationCalculator.hpp"
#include <iostream>
#include <algorithm>

Eigen::MatrixXd CorrelationCalculator::convolve(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel) {
    Eigen::MatrixXd kernel_rot = rotate180(kernel);
    return crossCorrelate(subject, kernel_rot);
}

Eigen::MatrixXd CorrelationCalculator::crossCorrelate(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel, int output_rows, int output_cols) {
    if (output_rows == -1) output_rows = subject.rows();
    if (output_cols == -1) output_cols = subject.cols();
    int pad_rows_total = std::max((long) 0, (output_rows - 1) + kernel.rows() - subject.rows());
    int pad_cols_total = std::max((long) 0, (output_cols - 1) + kernel.cols() - subject.cols());

    int pad_top = pad_rows_total / 2;
    int pad_bottom = pad_rows_total - pad_top;
    int pad_left = pad_cols_total / 2;
    int pad_right = pad_cols_total - pad_left;


    Eigen::MatrixXd padded_subject = Eigen::MatrixXd::Zero(subject.rows() + pad_rows_total, subject.cols() + pad_cols_total);
    padded_subject.block(pad_top, pad_left, subject.rows(), subject.cols()) = subject;

    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(output_rows, output_cols);
    Eigen::VectorXd kernel_vector = kernel.reshaped();
    for (int i = 0; i < output_rows; i++)
    {
        for (int j = 0; j < output_cols; j++)
        {
            Eigen::MatrixXd region = padded_subject.block(i, j, kernel.rows(), kernel.cols());
            Eigen::VectorXd region_vector = region.reshaped();
            output(i, j) = region_vector.dot(kernel_vector);
        }
        
    }
    
    return output;
}

Eigen::MatrixXd CorrelationCalculator::rotate180(const Eigen::MatrixXd &matrix) {
    return matrix.colwise().reverse().rowwise().reverse();
}