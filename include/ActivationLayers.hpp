/**
 * This file holds class definitions for several ActivationLayers using different activation functions,
 * primarily for convenience.
 */
#ifndef ACTTIVATION_LAYERS_HPP
#define ACTTIVATION_LAYERS_HPP

#include "ActivationLayer.hpp"
#include "ActivationFunctions.hpp"

class ReLU : public ActivationLayer
{
    ReLU() : ActivationLayer(ActivationFunctions::relu, ActivationFunctions::relu_prime) {}
};

class Tanh : public ActivationLayer
{
    Tanh() : ActivationLayer(ActivationFunctions::tanh, ActivationFunctions::tanh_prime) {}
};

class Sigmoid : public ActivationLayer
{
    Sigmoid() : ActivationLayer(ActivationFunctions::sigmoid, ActivationFunctions::sigmoid_prime) {}
};

#endif // ACTTIVATION_LAYERS_HPP
