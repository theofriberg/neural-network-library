#include "../include/CorrelationCalculator.hpp"
#include <algorithm>
#include <iostream>

MatrixXdRM CorrelationCalculator::convolve(const MatrixXdRM &subject, const MatrixXdRM &kernel)
{
    MatrixXdRM kernel_rot = rotate180(kernel);
    return crossCorrelate(subject, kernel_rot);
}

MatrixXdRM CorrelationCalculator::crossCorrelate(const MatrixXdRM &subject,
                                                 const MatrixXdRM &kernel, int output_rows,
                                                 int output_cols)
{
    if (output_rows == -1)
        output_rows = subject.rows();
    if (output_cols == -1)
        output_cols = subject.cols();
    int pad_rows_total = std::max((long)0, (output_rows - 1) + kernel.rows() - subject.rows());
    int pad_cols_total = std::max((long)0, (output_cols - 1) + kernel.cols() - subject.cols());

    int pad_top = pad_rows_total / 2;
    int pad_bottom = pad_rows_total - pad_top;
    int pad_left = pad_cols_total / 2;
    int pad_right = pad_cols_total - pad_left;

    MatrixXdRM padded_subject =
        MatrixXdRM::Zero(subject.rows() + pad_rows_total, subject.cols() + pad_cols_total);
    padded_subject.block(pad_top, pad_left, subject.rows(), subject.cols()) = subject;

    MatrixXdRM output = MatrixXdRM::Zero(output_rows, output_cols);
    VectorXd kernel_vector = kernel.reshaped();
    for (int i = 0; i < output_rows; i++)
    {
        for (int j = 0; j < output_cols; j++)
        {
            MatrixXdRM region = padded_subject.block(i, j, kernel.rows(), kernel.cols());
            VectorXd region_vector = region.reshaped();
            output(i, j) = region_vector.dot(kernel_vector);
        }
    }

    return output;
}

MatrixXdRM CorrelationCalculator::rotate180(const MatrixXdRM &matrix)
{
    return matrix.colwise().reverse().rowwise().reverse();
}