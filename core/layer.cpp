#include "layer.h"
#include <random>

namespace nn
{

layer::layer(const std::size_t &nr_size, const std::size_t &lr_size) : _b(1, lr_size), _w(nr_size, lr_size)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);

    for (size_t i = 0; i < lr_size; ++i)
        for (size_t j = 0; j < nr_size; ++j)
            _w.set(i, j, distribution(generator));

    for (size_t i = 0; i < lr_size; ++i)
        _b.set(i, 0, distribution(generator));
}

layer::~layer() {}

matrix layer::forward(const matrix &input) { return _w * input + _b; }
} // namespace nn
