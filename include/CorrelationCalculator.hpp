#ifndef CORRELATION_CALCULATOR_HPP
#define CORRELATION_CALCULATOR_HPP

#include <Eigen/Dense>

class CorrelationCalculator {

    public:
        static Eigen::MatrixXd crossCorrelate(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel, int output_rows = -1, int output_cols = -1);
        static Eigen::MatrixXd convolve(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel);
    
    private:
        static Eigen::MatrixXd rotate180(const Eigen::MatrixXd &matrix);
};


#endif // CORRELATION_CALCULATOR_HPP