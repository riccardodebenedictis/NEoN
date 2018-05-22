#include "activation_function.h"
#include <math.h>

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

double sigmoid::derivative(const double &val) const { return val * (1.0 - val); }

} // namespace nn
