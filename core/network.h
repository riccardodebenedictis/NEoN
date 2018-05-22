#pragma once

#include "error_function.h"
#include "activation_function.h"
#include <vector>
#include <cstddef>

namespace nn
{

class neuron
{
  friend class network;

private:
  activation_function &act_f;
  std::vector<double> weights;
  double bias;
  double output;

public:
  const size_t size;

public:
  neuron(activation_function &af, const size_t &size);
  ~neuron();

  double forward(const std::vector<double> &input);
};

class layer
{
  friend class network;

private:
  std::vector<neuron *> neurons;

public:
  const size_t size;

public:
  layer(activation_function &af, const size_t &lr_size, const size_t &nr_size);
  ~layer();

  std::vector<double> forward(const std::vector<double> &input);
};

class network
{
private:
  error_function &error_f;
  std::vector<layer *> layers;

public:
  network(error_function &ef, activation_function &af, const std::vector<size_t> &sizes);
  ~network();

  std::vector<double> forward(const std::vector<double> &input);
};
} // namespace nn