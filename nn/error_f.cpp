#include "error_f.h"
#include <cmath>
#include <cassert>

namespace nn
{

double mean_squared_error::error(const std::vector<double> &a, const std::vector<double> &y) const
{
    assert(a.size() == y.size());
    double err = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        err += pow(a[i] - y[i], 2);
    return err;
}

std::vector<double> mean_squared_error::delta(const activation_f &af, const std::vector<double> &z, const std::vector<double> &a, const std::vector<double> &y) const
{
    assert(z.size() == a.size());
    assert(a.size() == y.size());
    std::vector<double> d(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        d[i] = (a[i] - y[i]) * af.derivative(z[i]);
    return d;
}

double cross_entropy::error(const std::vector<double> &a, const std::vector<double> &y) const
{
    assert(a.size() == y.size());
    double err = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        err += y[i] * log(a[i]) + (1 - y[i]) * log(1 - a[i]);
    return isnan(err) ? 0 : -err;
}

std::vector<double> cross_entropy::delta(const activation_f &af, const std::vector<double> &z, const std::vector<double> &a, const std::vector<double> &y) const
{
    assert(z.size() == a.size());
    assert(a.size() == y.size());
    std::vector<double> d(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        d[i] = a[i] - y[i];
    return d;
}
} // namespace nn