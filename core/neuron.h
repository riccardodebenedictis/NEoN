#include "activation_function.h"
#include <cstddef>
#include <vector>

namespace nn
{

class neuron
{
  public:
    neuron(const std::size_t &size);
    ~neuron();

    double forward(const std::vector<double> &input, const activation_function &af) const;

  private:
    std::vector<double> weights;
    double bias;
};
} // namespace nn