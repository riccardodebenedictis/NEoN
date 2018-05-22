#include "network.h"
#include "error_function.h"
#include "activation_function.h"
#include <random>

namespace nn
{

neuron::neuron(activation_function &af, const size_t &size) : act_f(af), weights(std::vector<double>(size)), size(size)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);

    for (size_t i = 0; i < size; ++i)
        weights[i] = distribution(generator);
    bias = distribution(generator);
}

neuron::~neuron() {}

double neuron::forward(const std::vector<double> &input)
{
    output = bias;
    for (size_t i = 0; i < size; ++i)
        output += weights[i] * input[i];
    output = act_f.compute(output);
    return output;
}

layer::layer(activation_function &af, const std::size_t &lr_size, const std::size_t &nr_size) : neurons(lr_size), size(lr_size)
{
    for (size_t i = 0; i < size; ++i)
        neurons[i] = new neuron(af, nr_size);
}

layer::~layer()
{
    for (size_t i = 0; i < size; ++i)
        delete neurons[i];
}

std::vector<double> layer::forward(const std::vector<double> &input)
{
    std::vector<double> output(neurons.size());
    for (size_t i = 0; i < size; ++i)
        output[i] = neurons[i]->forward(input);
    return output;
}

network::network(error_function &ef, activation_function &af, const std::vector<size_t> &sizes) : error_f(ef), layers(sizes.size() - 1)
{
    for (int i = 0; i < sizes.size() - 1; i++)
        layers[i] = new layer(af, sizes[i + 1], sizes[i]);
    for (neuron *n : layers[0]->neurons)
        n->bias = 0;
}

network::~network()
{
    for (size_t i = 0; i < layers.size(); ++i)
        delete layers[i];
}

std::vector<double> network::forward(const std::vector<double> &input)
{
    std::vector<double> output = layers[0]->forward(input);
    for (int i = 1; i < layers.size(); i++)
        output = layers[i]->forward(output);
    return output;
}
} // namespace nn