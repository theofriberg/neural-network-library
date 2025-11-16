#include "../include/CorrelationCalculator.hpp"
#include <iostream>

Eigen::MatrixXd CorrelationCalculator::convolve(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel) {
    Eigen::MatrixXd kernel_rot = rotate180(kernel);
    return crossCorrelate(subject, kernel_rot);
}

Eigen::MatrixXd CorrelationCalculator::crossCorrelate(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel) {
    int kernel_rows = kernel.rows();
    int kernel_cols = kernel.cols();
    int subject_rows = subject.rows();
    int subject_cols = subject.cols();
    if (kernel_rows > subject_rows || kernel_cols > subject_cols)
    {
        throw std::invalid_argument("Kernel dimension must be less than or equal to subject dimension.");
    }

    int pad_vert = kernel_rows - 1;
    int pad_horizontal = kernel_cols - 1;
    Eigen::MatrixXd padded_subject = Eigen::MatrixXd::Zero(subject_rows + 2*pad_vert, subject_cols + 2*pad_horizontal);
    padded_subject.block(pad_vert, pad_horizontal, subject_rows, subject_cols) = subject;
    
    int output_rows = subject_rows + kernel_rows - 1;
    int output_cols = subject_cols + kernel_cols - 1;
    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(output_rows, output_cols);
    for (int i = 0; i < output_rows; i++)
    {
        for (int j = 0; j < output_cols; j++)
        {
            Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>> subject_patch(
                padded_subject.data() + i + j * padded_subject.rows(),
                kernel_rows,
                kernel_cols,
                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(padded_subject.rows(), 1)
            );
            output(i, j) = (subject_patch.array() * kernel.array()).sum();
        }
    }
    
    return output;
}

Eigen::MatrixXd CorrelationCalculator::rotate180(const Eigen::MatrixXd &matrix) {
    return matrix.colwise().reverse().rowwise().reverse();
}