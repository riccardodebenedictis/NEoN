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

public:
  activation_function &act_f;
  const size_t size;

private:
  std::vector<double> weights;
  double bias;
  double output;
  double delta;
  std::vector<double> nabla_w;
  double nabla_b;

public:
  neuron(activation_function &af, const size_t &size);
  ~neuron();

  double forward(const std::vector<double> &input);
  double get_weight(const size_t &w) const { return weights[w]; }
  double get_bias() const { return bias; }
  double get_delta() const { return delta; }
  double get_output() const { return output; }
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

  neuron &get_neuron(const size_t &n) const { return *neurons[n]; }

  std::vector<double> forward(const std::vector<double> &input);
};

class network
{
  friend void error_function::set_delta(network &net, const size_t &l, const size_t &n, const double &delta);

private:
  error_function &error_f;
  std::vector<layer *> layers;

public:
  const size_t size;

public:
  network(error_function &ef, activation_function &af, const std::vector<size_t> &sizes);
  ~network();

  layer &get_layer(const size_t &l) const { return *layers[l]; }

  std::vector<double> forward(const std::vector<double> &input);

  void sgd(std::vector<training_data *> &data, const size_t &epochs, const size_t &mini_batch_size, const double &eta);

private:
  void set_delta(const size_t &l, const size_t &n, const double &delta) { layers[l]->neurons[n]->delta = delta; }

  void update_mini_batch(const std::vector<training_data *> &mini_batch, const double &eta);
  void backprop(const training_data &data);
};
} // namespace nn