#include "network.h"
#include "error_function.h"
#include "activation_function.h"
#ifndef NDEBUG
#include "network_listener.h"
#endif
#include <random>
#include <algorithm>

namespace nn
{

neuron::neuron(activation_function &af, const std::size_t &size) : act_f(af), size(size), weights(std::vector<double>(size)), nabla_w(std::vector<double>(size))
{
    std::default_random_engine gen;
    std::normal_distribution<double> distribution(0, 1);

    for (std::size_t i = 0; i < size; ++i)
        weights[i] = distribution(gen);
    bias = distribution(gen);
}

neuron::~neuron() {}

double neuron::forward(const std::vector<double> &input)
{
    output = bias;
    for (std::size_t i = 0; i < size; ++i)
        output += weights[i] * input[i];
    output = act_f.compute(output);
    return output;
}

layer::layer(activation_function &af, const std::size_t &lr_size, const std::size_t &nr_size) : neurons(lr_size), size(lr_size)
{
    for (std::size_t i = 0; i < size; ++i)
        neurons[i] = new neuron(af, nr_size);
}

layer::~layer()
{
    for (std::size_t i = 0; i < size; ++i)
        delete neurons[i];
}

std::vector<double> layer::forward(const std::vector<double> &input)
{
    std::vector<double> output(neurons.size());
    for (std::size_t i = 0; i < size; ++i)
        output[i] = neurons[i]->forward(input);
    return output;
}

network::network(error_function &ef, activation_function &af, const std::vector<std::size_t> &sizes) : error_f(ef), layers(sizes.size() - 1), size(sizes.size() - 1)
{
    for (std::size_t i = 0; i < sizes.size() - 1; ++i)
        layers[i] = new layer(af, sizes[i + 1], sizes[i]);
    for (neuron *n : layers[0]->neurons)
        n->bias = 0;
}

network::~network()
{
    for (std::size_t i = 0; i < layers.size(); ++i)
        delete layers[i];
}

std::vector<double> network::forward(const std::vector<double> &input)
{
    std::vector<double> output = layers[0]->forward(input);
    for (std::size_t i = 1; i < layers.size(); ++i)
        output = layers[i]->forward(output);
    return output;
}

void network::sgd(std::vector<training_data *> &data, const std::size_t &epochs, const std::size_t &mini_batch_size, const double &eta)
{
#ifndef NDEBUG
    // we notify the listeners that we are starting a training phase..
    for (const auto &l : listeners)
        l->start_training(get_error(data));
#endif
    std::default_random_engine gen;
    for (std::size_t i = 1; i <= epochs; ++i)
    {
#ifndef NDEBUG
        // we notify the listeners that we are starting a new epoch..
        for (const auto &l : listeners)
            l->start_epoch(get_error(data));
#endif
        // we shuffle the training data..
        std::shuffle(data.begin(), data.end(), gen);
        // we partition the training data into mini batches of 'mini_batch_size' size..
        for (std::size_t j = 0; j <= data.size() - mini_batch_size; j += mini_batch_size)
            update_mini_batch(std::vector<training_data *>(data.begin() + j, data.begin() + j + mini_batch_size), eta);
#ifndef NDEBUG
        // we notify the listeners that we have finished an epoch..
        for (const auto &l : listeners)
            l->stop_epoch(get_error(data));
#endif
    }
#ifndef NDEBUG
    // we notify the listeners that we have finished a training phase..
    for (const auto &l : listeners)
        l->stop_training(get_error(data));
#endif
}

void network::update_mini_batch(const std::vector<training_data *> &mini_batch, const double &eta)
{
    // we perform backpropagation..
    for (training_data *data : mini_batch)
        backprop(*data);

    // we update the biases, the weigths, and clean up things..
    for (std::size_t i = 0; i < size; ++i)
        for (std::size_t j = 0; j < layers[i]->size; ++j)
        {
            neuron &n = *layers[i]->neurons[j];
            n.bias -= (eta / mini_batch.size()) * n.nabla_b;
            n.nabla_b = 0;
            for (std::size_t k = 0; k < n.size; ++k)
                n.weights[k] -= (eta / mini_batch.size()) * n.nabla_w[k];
            n.nabla_w.assign(n.size, 0);
        }
}

void network::backprop(const training_data &data)
{
    error_f.compute_deltas(*this, data);

    // we use the computed deltas to update the nablas..
    for (std::size_t i = size - 1; i >= 1; --i)
    {
        layer &l = *layers[i];
        layer &l_prev = *layers[i - 1];
        for (std::size_t j = 0; j < l.size; ++j)
        {
            neuron &n = *l.neurons[j];
            n.nabla_b += n.delta;
            for (std::size_t k = 0; k < l_prev.size; ++k)
                n.nabla_w[k] += l_prev.neurons[k]->output * n.delta;
        }
    }
    layer &l0 = *layers[0];
    for (std::size_t i = 0; i < l0.size; i++)
    {
        neuron &n = *l0.neurons[i];
        n.nabla_b += n.delta;
        for (std::size_t k = 0; k < data.input.size(); ++k)
            n.nabla_w[k] += data.input[k] * n.delta;
    }
}
} // namespace nn