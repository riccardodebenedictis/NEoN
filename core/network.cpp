#include "network.h"
#include "error_function.h"
#include "activation_function.h"
#ifndef NDEBUG
#include "network_listener.h"
#endif
#include <algorithm>

namespace nn
{

neuron::neuron(std::default_random_engine &gen, activation_function &af, const std::size_t &size) : act_f(af), size(size), weights(std::vector<double>(size)), nabla_w(std::vector<double>(size, 0.0)), nabla_b(0)
{
    std::normal_distribution<double> distribution(0, 1);

    for (std::size_t i = 0; i < size; ++i)
        weights[i] = distribution(gen) / std::sqrt(size);
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

layer::layer(std::default_random_engine &gen, activation_function &af, const std::size_t &lr_size, const std::size_t &nr_size) : neurons(lr_size), size(lr_size)
{
    for (std::size_t i = 0; i < size; ++i)
        neurons[i] = new neuron(gen, af, nr_size);
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
        layers[i] = new layer(gen, af, sizes[i + 1], sizes[i]);
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

void network::sgd(std::vector<data_row *> &tr_data, std::vector<data_row *> &tst_data, const std::size_t &epochs, const std::size_t &mini_batch_size, const double &eta, const double &lambda)
{
#ifndef NDEBUG
    // we notify the listeners that we are starting a training phase..
    for (const auto &l : listeners)
        l->start_training(epochs, get_error(tr_data), get_error(tst_data));
#endif
    for (std::size_t i = 1; i <= epochs; ++i)
    {
#ifndef NDEBUG
        // we notify the listeners that we are starting a new epoch..
        for (const auto &l : listeners)
            l->start_epoch(get_error(tr_data), get_error(tst_data));
#endif
        // we shuffle the training data..
        std::shuffle(tr_data.begin(), tr_data.end(), gen);
        // we partition the training data into mini batches of 'mini_batch_size' size..
        for (std::size_t j = 0; j <= tr_data.size() - mini_batch_size; j += mini_batch_size)
            update_mini_batch(std::vector<data_row *>(tr_data.begin() + j, tr_data.begin() + j + mini_batch_size), eta, lambda);
#ifndef NDEBUG
        // we notify the listeners that we have finished an epoch..
        for (const auto &l : listeners)
            l->stop_epoch(get_error(tr_data), get_error(tst_data));
#endif
    }
#ifndef NDEBUG
    // we notify the listeners that we have finished a training phase..
    for (const auto &l : listeners)
        l->stop_training(get_error(tr_data), get_error(tst_data));
#endif
}

void network::update_mini_batch(const std::vector<data_row *> &mini_batch, const double &eta, const double &lambda)
{
    // we perform backpropagation..
    for (data_row *data : mini_batch)
        backprop(*data);

    // we update the biases, the weigths, and clean up things..
    for (std::size_t i = 0; i < size; ++i)
        for (std::size_t j = 0; j < layers[i]->size; ++j)
        {
            neuron &n = *layers[i]->neurons[j];
            n.bias -= (eta / mini_batch.size()) * n.nabla_b;
            n.nabla_b = 0;
            for (std::size_t k = 0; k < n.size; ++k)
                n.weights[k] -= (eta / mini_batch.size()) * n.nabla_w[k] + ((eta * lambda) / mini_batch.size()) * n.weights[k];
            n.nabla_w.assign(n.size, 0);
        }
}

void network::backprop(const data_row &data)
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

#ifndef NDEBUG
void network::add_listener(network_listener &l)
{
    listeners.push_back(&l);
}
void network::remove_listener(network_listener &l)
{
    listeners.erase(std::find(listeners.begin(), listeners.end(), &l));
}
#endif
} // namespace nn