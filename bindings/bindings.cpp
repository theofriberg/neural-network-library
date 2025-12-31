#include "ActivationFunctions.hpp"
#include "ActivationLayers.hpp"
#include "ConvolutionalLayer.hpp"
#include "DenseLayer.hpp"
#include "LossFunctions.hpp"
#include "Network.hpp"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::shared_ptr<Network> makeMseNet()
{
    return std::make_shared<Network>(LossFunctions::mse, LossFunctions::msePrime);
}

std::shared_ptr<Network> makeCCENet()
{
    return std::make_shared<Network>(LossFunctions::categoricalCrossentropy,
                                     LossFunctions::categoricalCrossentropyPrime);
}

PYBIND11_MODULE(mynn, m)
{
    m.doc() = "Python module for the neural-network-library";

    py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer");

    py::class_<Network, std::shared_ptr<Network>>(m, "Network")
        .def_static("mse", &makeMseNet)
        .def_static("categorical_cross_entropy", &makeCCENet)
        .def("train", &Network::train)
        .def("predict", &Network::predict)
        .def("add_layer", &Network::addLayer);

    py::class_<DenseLayer, Layer, std::shared_ptr<DenseLayer>>(m, "DenseLayer")
        .def(py::init<int, int>())
        .def("backward", &DenseLayer::backward)
        .def("forward", &DenseLayer::forward);

    py::class_<ConvolutionalLayer, Layer, std::shared_ptr<ConvolutionalLayer>>(m, "ConvLayer")
        .def(py::init<int, int, int, int>())
        .def("backward", &ConvolutionalLayer::backward)
        .def("forward", &ConvolutionalLayer::forward);

    py::class_<ActivationLayer, Layer, std::shared_ptr<ActivationLayer>>(m, "ActivationLayer")
        .def("backward", &ActivationLayer::backward)
        .def("forward_batch", &ActivationLayer::forward);

    py::class_<ReLU, ActivationLayer, std::shared_ptr<ReLU>>(m, "ReLU").def(py::init<>());

    py::class_<Tanh, ActivationLayer, std::shared_ptr<Tanh>>(m, "Tanh").def(py::init<>());

    py::class_<Sigmoid, ActivationLayer, std::shared_ptr<Sigmoid>>(m, "Sigmoid").def(py::init<>());
}