#pragma once

#include <vector>

namespace nn
{

class network;

class data_row
{
public:
  const std::vector<double> input;
  const std::vector<double> output;

public:
  data_row(const std::vector<double> &input, const std::vector<double> &output) : input(input), output(output) {}
  data_row(const data_row &orig) = delete;
  ~data_row() {}
};

class error_function
{
  friend class network;

public:
  error_function() {}
  error_function(const error_function &orig) = delete;
  ~error_function() {}

private:
  virtual double error(network &net, const std::vector<data_row *> &data) = 0;
  virtual void compute_deltas(network &net, const data_row &data) = 0;

protected:
  inline void set_delta(network &net, const std::size_t &l, const std::size_t &n, const double &delta);
};

class mean_squared_error : public error_function
{
public:
  mean_squared_error() {}
  mean_squared_error(const mean_squared_error &orig) = delete;
  ~mean_squared_error() {}

  double error(network &net, const std::vector<data_row *> &data) override;
  void compute_deltas(network &net, const data_row &data) override;
};

class cross_entropy : public error_function
{
public:
  cross_entropy() {}
  cross_entropy(const cross_entropy &orig) = delete;
  ~cross_entropy() {}

  double error(network &net, const std::vector<data_row *> &data) override;
  void compute_deltas(network &net, const data_row &data) override;
};
} // namespace nn