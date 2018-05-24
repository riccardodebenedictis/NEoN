#pragma once

#include "activation_f.h"
#include <vector>

namespace nn
{

class data_row
{
public:
  const std::vector<double> x;
  const std::vector<double> y;

public:
  data_row(const std::vector<double> &x, const std::vector<double> &y) : x(x), y(y) {}
  data_row(const data_row &orig) = delete;
  ~data_row() {}
};

class error_f
{

public:
  error_f() {}
  error_f(const error_f &orig) = delete;
  ~error_f() {}

  virtual double error(const std::vector<double> &a, const std::vector<double> &y) const = 0;
  virtual std::vector<double> delta(const activation_f &af, const std::vector<double> &z, const std::vector<double> &a, const std::vector<double> &y) const = 0;
};

class mean_squared_error : public error_f
{

public:
  mean_squared_error() {}
  mean_squared_error(const mean_squared_error &orig) = delete;
  ~mean_squared_error() {}

  double error(const std::vector<double> &a, const std::vector<double> &y) const override;
  std::vector<double> delta(const activation_f &af, const std::vector<double> &z, const std::vector<double> &a, const std::vector<double> &y) const override;
};

class cross_entropy : public error_f
{

public:
  cross_entropy() {}
  cross_entropy(const cross_entropy &orig) = delete;
  ~cross_entropy() {}

  double error(const std::vector<double> &a, const std::vector<double> &y) const override;
  std::vector<double> delta(const activation_f &af, const std::vector<double> &z, const std::vector<double> &a, const std::vector<double> &y) const override;
};
} // namespace nn