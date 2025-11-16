#ifndef CORRELATION_CALCULATOR_HPP
#define CORRELATION_CALCULATOR_HPP

#include "Eigen/Dense"

class CorrelationCalculator {

    public:
        static Eigen::MatrixXd crossCorrelate(Eigen::MatrixXd &subject, Eigen::MatrixXd &kernel);
        static Eigen::MatrixXd convolve(Eigen::MatrixXd &subject, Eigen::MatrixXd &kernel);
    
    private:
        static Eigen::MatrixXd rotate180(Eigen::MatrixXd &matrix);
};


#endif // CORRELATION_CALCULATOR_HPP