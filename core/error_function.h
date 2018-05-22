#pragma once

#include <vector>

namespace nn
{

class network;

class training_data
{
public:
  const std::vector<double> input;
  const std::vector<double> output;

public:
  training_data(const std::vector<double> &input, const std::vector<double> &output) : input(input), output(output) {}
  training_data(const training_data &orig) = delete;
  ~training_data() {}
};

class error_function
{
  friend class network;

public:
  error_function() {}
  error_function(const error_function &orig) = delete;
  ~error_function() {}

  virtual double error(network &net, const std::vector<training_data *> &data) = 0;
  virtual void compute_deltas(network &net, const training_data &data) = 0;

protected:
  inline void set_delta(network &net, const std::size_t &l, const std::size_t &n, const double &delta);
};

class mean_squared_error : public error_function
{
public:
  mean_squared_error() {}
  mean_squared_error(const mean_squared_error &orig) = delete;
  ~mean_squared_error() {}

  double error(network &net, const std::vector<training_data *> &data) override;
  void compute_deltas(network &net, const training_data &data) override;
};

class cross_entropy : public error_function
{
public:
  cross_entropy() {}
  cross_entropy(const cross_entropy &orig) = delete;
  ~cross_entropy() {}

  double error(network &net, const std::vector<training_data *> &data) override;
  void compute_deltas(network &net, const training_data &data) override;
};
} // namespace nn