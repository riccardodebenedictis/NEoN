#include "error_function.h"
#include "network.h"

namespace nn
{
double mean_squared_error::error(network &net, std::vector<training_data *> &data)
{
    double err = 0;
    for (training_data *d : data)
    {
        std::vector<double> c_output = net.forward(d->input);
        for (int i = 0; i < c_output.size(); i++)
            err += pow(d->output[i] - c_output[i], 2);
    }
    return err / data.size();
}

void mean_squared_error::compute_deltas(network &net, training_data &data) {}

double cross_entropy::error(network &net, std::vector<training_data *> &data)
{
    double err = 0;
    for (training_data *d : data)
    {
        std::vector<double> c_output = net.forward(d->input);
        for (int i = 0; i < c_output.size(); i++)
            err += d->output[i] * log(c_output[i]) + (1 - d->output[i]) * log(1 - c_output[i]);
    }
    return -err / data.size();
}

void cross_entropy::compute_deltas(network &net, training_data &data) {}
} // namespace nn