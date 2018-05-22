#include "error_function.h"

namespace nn
{
double mean_squared_error::error(network &net, std::vector<training_data *> &data)
{
    return 0;
}

void mean_squared_error::compute_deltas(network &net, training_data &data) {}
} // namespace nn

/*
#include "network.h"
#include <cmath>

namespace nn
{

double mean_squared_error::error(network &net, std::vector<training_data *> &data)
{
    double cost = 0;
    for (training_data *d : data)
    {
        std::vector<double> c_output = net.forward(d->input);
        for (int i = 0; i < c_output.size(); i++)
            cost += pow(d->output[i] - c_output[i], 2);
    }
    return cost / data.size();
}

void mean_squared_error::compute_deltas(network &net, training_data &data)
{
}
} // namespace nn
*/