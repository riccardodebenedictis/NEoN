#include "error_function.h"
#include "network.h"
#include <cmath>

namespace nn
{

inline void error_function::set_delta(network &net, const std::size_t &l, const std::size_t &n, const double &delta) { net.set_delta(l, n, delta); }

double mean_squared_error::error(network &net, const std::vector<data_row *> &data)
{
    double err = 0;
    for (data_row *d : data)
    {
        std::vector<double> c_output = net.forward(d->input);
        for (std::size_t i = 0; i < c_output.size(); ++i)
            err += pow(d->output[i] - c_output[i], 2);
    }
    return err / data.size();
}

void mean_squared_error::compute_deltas(network &net, const data_row &data)
{
    // forward propagation..
    std::vector<double> c_output = net.forward(data.input);

    // back propagation..
    // we compute the deltas for the output layer..
    for (std::size_t i = 0; i < data.output.size(); ++i)
    {
        neuron &n = net.get_layer(net.size - 1).get_neuron(i);
        set_delta(net, net.size - 1, i, -(data.output[i] - c_output[i]) * n.act_f.derivative(n.get_output()));
    }

    // we compute the deltas for the other layers..
    for (std::size_t i = net.size - 1; i > 0; --i)
    {
        layer &l = net.get_layer(i - 1);
        layer &l_next = net.get_layer(i);

        for (std::size_t j = 0; j < l.size; ++j)
        {
            double delta = 0;
            for (std::size_t k = 0; k < l_next.size; ++k)
                delta += l_next.get_neuron(k).get_weight(j) * l_next.get_neuron(k).get_delta();
            delta *= l.get_neuron(j).act_f.derivative(l.get_neuron(j).get_output());
            set_delta(net, i - 1, j, delta);
        }
    }
}

double cross_entropy::error(network &net, const std::vector<data_row *> &data)
{
    double err = 0;
    for (data_row *d : data)
    {
        std::vector<double> c_output = net.forward(d->input);
        for (std::size_t i = 0; i < c_output.size(); ++i)
            err += d->output[i] * log(c_output[i]) + (1 - d->output[i]) * log(1 - c_output[i]);
    }
    return -err / data.size();
}

void cross_entropy::compute_deltas(network &net, const data_row &data)
{
    // forward propagation..
    std::vector<double> c_output = net.forward(data.input);

    // back propagation..
    // we compute the deltas for the output layer..
    for (std::size_t i = 0; i < data.output.size(); ++i)
        set_delta(net, net.size - 1, i, -(data.output[i] - c_output[i]));

    // we compute the deltas for the other layers..
    for (std::size_t i = net.size - 1; i > 0; --i)
    {
        layer &l = net.get_layer(i - 1);
        layer &l_next = net.get_layer(i);

        for (std::size_t j = 0; j < l.size; ++j)
        {
            double delta = 0;
            for (std::size_t k = 0; k < l_next.size; ++k)
                delta += l_next.get_neuron(k).get_weight(j) * l_next.get_neuron(k).get_delta();
            set_delta(net, i - 1, j, delta);
        }
    }
}
} // namespace nn