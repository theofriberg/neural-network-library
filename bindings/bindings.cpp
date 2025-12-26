#include "../include/ActivationFunctions.hpp"
#include "../include/ActivationLayers.hpp"
#include "../include/ConvolutionalLayer.hpp"
#include "../include/DenseLayer.hpp"
#include "../include/LossFunctions.hpp"
#include "../include/Network.hpp"
#include "../include/ReshapeLayer.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindLossFunctions(py::module_ &m)
{

    py::module_ loss = m.def_submodule("loss", "Contains loss functions with derivatives");

    loss.def("mse", LossFunctions::mse);
    loss.def("mse_prime", LossFunctions::msePrime);
    loss.def("categorical_cross_entropy", LossFunctions::categoricalCrossentropy);
    loss.def("categorical_cross_entropy_prime", LossFunctions::categoricalCrossentropyPrime);
}

PYBIND11_MODULE(mynn, m)
{
    m.doc() = "Python module for the neural-network-library";

    bindLossFunctions(m);

    py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer");

    py::class_<Network, std::shared_ptr<Network>>(m, "Network")
        .def(py::init<std::function<double(Eigen::MatrixXd, Eigen::MatrixXd)>,
                      std::function<Eigen::MatrixXd(Eigen::MatrixXd, Eigen::MatrixXd)>>())
        .def("train", &Network::train)
        .def("predict", &Network::predict)
        .def("add_layer", &Network::addLayer);

    py::class_<DenseLayer, Layer, std::shared_ptr<DenseLayer>>(m, "DenseLayer")
        .def(py::init<int, int>())
        .def("backward_batch", &DenseLayer::backwardBatch)
        .def("forward_batch", &DenseLayer::forwardBatch)
        .def("backward", &DenseLayer::backward)
        .def("forward", &DenseLayer::forward);

    py::class_<ConvolutionalLayer, Layer, std::shared_ptr<ConvolutionalLayer>>(m, "ConvLayer")
        .def(py::init<int, int>())
        .def("backward_batch", &ConvolutionalLayer::backwardBatch)
        .def("forward_batch", &ConvolutionalLayer::forwardBatch)
        .def("backward", &ConvolutionalLayer::backward)
        .def("forward", &ConvolutionalLayer::forward);

    py::class_<ActivationLayer, Layer, std::shared_ptr<ActivationLayer>>(m, "ActivationLayer")
        .def("backward_batch", &ActivationLayer::backwardBatch)
        .def("forward_batch", &ActivationLayer::forwardBatch)
        .def("backward", &ActivationLayer::backward)
        .def("forward", &ActivationLayer::forward);

    py::class_<ReLU, ActivationLayer, std::shared_ptr<ReLU>>(m, "ReLU").def(py::init<>());

    py::class_<Tanh, ActivationLayer, std::shared_ptr<Tanh>>(m, "Tanh").def(py::init<>());

    py::class_<Sigmoid, ActivationLayer, std::shared_ptr<Sigmoid>>(m, "Sigmoid").def(py::init<>());
}