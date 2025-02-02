/**
 * This file holds class definitions for several ActivationLayers using
 * different activation functions, primarily for convenience.
 */
#ifndef ACTTIVATION_LAYERS_HPP
#define ACTTIVATION_LAYERS_HPP

#include "ActivationFunctions.hpp"
#include "ActivationLayer.hpp"

class ReLU : public ActivationLayer {
public:
  ReLU()
      : ActivationLayer(ActivationFunctions::relu,
                        ActivationFunctions::reluPrime, "relu") {}
};

class Tanh : public ActivationLayer {
public:
  Tanh()
      : ActivationLayer(ActivationFunctions::tanh,
                        ActivationFunctions::tanhPrime, "tanh") {}
};

class Sigmoid : public ActivationLayer {
public:
  Sigmoid()
      : ActivationLayer(ActivationFunctions::sigmoid,
                        ActivationFunctions::sigmoidPrime, "sigmoid") {}
};

#endif // ACTTIVATION_LAYERS_HPP
