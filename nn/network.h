#pragma once

#include "error_f.h"
#ifndef NDEBUG
#include "network_listener.h"
#include <algorithm>
#endif
#include <random>

namespace nn
{

class layer
{
  friend class network;

private:
  std::vector<std::vector<double>> w;            // the weights of the layer..
  std::vector<double> b;                         // the biases of the layer..
  std::vector<double> z;                         // the weighted inputs to the neurons..
  std::vector<double> a;                         // the outputs of the neurons..
  std::vector<double> delta;                     // the delta errors of the neurons..
  std::vector<std::vector<double>> nabla_w;      // the partial derivative of the neurons..
  std::vector<double> nabla_b;                   // the derivative of the biases..
  std::vector<std::vector<double>> last_nabla_w; // the last partial derivative of the neurons (for momentum)..
  std::vector<double> last_nabla_b;              // the last derivative of the biases (for momentum)..

public:
  const std::size_t lr_size; // the number of neurons..
  const std::size_t nr_size; // the number of synapsis for each neuron..

public:
  layer(std::default_random_engine &gen, const std::size_t &lr_size, const std::size_t &nr_size);
  layer(const layer &orig) = delete;
  ~layer();

  std::vector<double> forward(const activation_f &af, const std::vector<double> &x);
};

class network
{

private:
  std::default_random_engine gen;
  error_f &ef;
  activation_f &af;
  std::vector<layer *> layers;

public:
  const std::size_t size;

public:
  network(error_f &ef, activation_f &af, const std::vector<std::size_t> &sizes);
  network(const network &orig) = delete;
  ~network();

  double get_error(std::vector<data_row *> &data);

  std::vector<double> forward(const std::vector<double> &x);

  /**
   * performs stochastic gradient descent on the given training data.
   * 
   * @param tr_data a vector of 'data_row's representing the training data.
   * @param eval_data a vector of 'data_row's representing the evaluation data (might be useful for meta-parameters initialization).
   * @param epochs a positive integer representing the number of epochs (i.e. the number of performed training steps).
   * @param mini_batch_size a positive integer representing the size of the mini-batch.
   * @param eta a positive real representing the learning rate.
   * @param mu a positive (and strictly less than one) real representing the momentum.
   * @param lambda a positive real representing the regularization parameter.
   */
  void sgd(std::vector<data_row *> &tr_data, std::vector<data_row *> &eval_data, const std::size_t &epochs, const std::size_t &mini_batch_size, const double &eta, const double &mu, const double &lambda);

private:
  void update_mini_batch(const std::vector<data_row *> &mini_batch, const double &eta, const double &mu, const double &lambda);
  void backprop(const data_row &data);

#ifndef NDEBUG
private:
  std::vector<network_listener *> listeners; // the network listeners..

public:
  void add_listener(network_listener &l) { listeners.push_back(&l); }
  void remove_listener(network_listener &l) { listeners.erase(std::find(listeners.begin(), listeners.end(), &l)); }
#endif
};
} // namespace nn