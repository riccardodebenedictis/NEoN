#include "activation_f.h"
#include <cmath>

namespace nn
{

double sigmoid::compute(const double &val) const
{
    if (val > 100)
        return 1.0;
    else if (val < -100)
        return 0.0;
    else
        return (1.0 / (1.0 + exp(-val)));
}
double sigmoid::derivative(const double &val) const { return compute(val) * (1.0 - compute(val)); }

double linear::compute(const double &val) const { return val; }
double linear::derivative(const double &val) const { return 1.0; }
} // namespace nn
