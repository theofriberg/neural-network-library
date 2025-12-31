#ifndef CORRELATION_CALCULATOR_HPP
#define CORRELATION_CALCULATOR_HPP

#include "Types.hpp"
#include <Eigen/Dense>

class CorrelationCalculator
{

  public:
    static MatrixXdRM crossCorrelate(const MatrixXdRM &subject, const MatrixXdRM &kernel,
                                     int output_rows = -1, int output_cols = -1);
    static MatrixXdRM convolve(const MatrixXdRM &subject, const MatrixXdRM &kernel);

  private:
    static MatrixXdRM rotate180(const MatrixXdRM &matrix);
};

#endif // CORRELATION_CALCULATOR_HPP