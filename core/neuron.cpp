#include "neuron.h"
#include <random>

namespace nn
{

neuron::neuron(const std::size_t &size)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);

    for (size_t i = 0; i < size; ++i)
        weights[i] = distribution(generator);
    bias = distribution(generator);
}

neuron::~neuron() {}
} // namespace nn