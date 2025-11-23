#ifndef CORRELATION_CALCULATOR_HPP
#define CORRELATION_CALCULATOR_HPP

#include <Eigen/Dense>

class CorrelationCalculator {

    public:
        static Eigen::MatrixXd sameCrossCorrelate(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel);
        static Eigen::MatrixXd validCrossCorrelate(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel);
        static Eigen::MatrixXd convolve(const Eigen::MatrixXd &subject, const Eigen::MatrixXd &kernel);
    
    private:
        static Eigen::MatrixXd rotate180(const Eigen::MatrixXd &matrix);
};


#endif // CORRELATION_CALCULATOR_HPP