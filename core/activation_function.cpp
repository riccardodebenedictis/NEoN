#include "activation_function.h"
#include <math.h>

namespace nn
{

double sigmoid::compute(const double &val)
{
    if (val > 100)
        output = 1.0;
    else if (val < -100)
        output = 0.0;
    else
        output = (1.0 / (1.0 + exp(-val)));
    return output;
}

double sigmoid::derivative() { return output * (1.0 - output); }

} // namespace nn
