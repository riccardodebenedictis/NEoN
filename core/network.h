#pragma once

#include "error_function.h"
#include "activation_function.h"
#include <random>
#include <vector>
#include <cstddef>

namespace nn
{

#ifndef NDEBUG
class network_listener;
#endif

class neuron
{
  friend class network;

public:
  activation_function &act_f; // the activation function of the neuron..
  const std::size_t size;     // the number of synapsis of the neuron..

private:
  std::vector<double> weights; // the weights of the neuron..
  double bias;                 // the bias..
  double output;               // the output computed by the last call to the 'forward' procedure..
  double delta;                // the delta error of the neuron..
  std::vector<double> nabla_w; // the partial derivatives over the weigths..
  double nabla_b;              // the partial derivative over the bias..

public:
  neuron(std::default_random_engine &gen, activation_function &af, const std::size_t &size);
  neuron(const neuron &orig) = delete;
  ~neuron();

  double forward(const std::vector<double> &input);
  double get_weight(const std::size_t &w) const { return weights[w]; }
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
  const std::size_t size;

public:
  layer(std::default_random_engine &gen, activation_function &af, const std::size_t &lr_size, const std::size_t &nr_size);
  layer(const layer &orig) = delete;
  ~layer();

  neuron &get_neuron(const std::size_t &n) const { return *neurons[n]; }

  std::vector<double> forward(const std::vector<double> &input);
};

class network
{
  friend void error_function::set_delta(network &net, const std::size_t &l, const std::size_t &n, const double &delta);

private:
  std::default_random_engine gen;
  error_function &error_f;
  std::vector<layer *> layers;
#ifndef NDEBUG
private:
  std::vector<network_listener *> listeners; // the network listeners..
#endif

public:
  const std::size_t size;

public:
  network(error_function &ef, activation_function &af, const std::vector<std::size_t> &sizes);
  network(const network &orig) = delete;
  ~network();

  layer &get_layer(const std::size_t &l) const { return *layers[l]; }

  std::vector<double> forward(const std::vector<double> &input);

  void sgd(std::vector<data_row *> &tr_data, std::vector<data_row *> &tst_data, const std::size_t &epochs, const std::size_t &mini_batch_size, const double &eta, const double &lambda);
  double get_error(const std::vector<data_row *> &data) { return error_f.error(*this, data); }

private:
  void set_delta(const std::size_t &l, const std::size_t &n, const double &delta) { layers[l]->neurons[n]->delta = delta; }

  void update_mini_batch(const std::vector<data_row *> &mini_batch, const double &eta, const double &lambda);
  void backprop(const data_row &data);

#ifndef NDEBUG
public:
  void add_listener(network_listener &l);
  void remove_listener(network_listener &l);
#endif
};
} // namespace nn