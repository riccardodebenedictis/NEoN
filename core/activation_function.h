#pragma once

namespace nn
{

class activation_function
{
public:
  activation_function() {}
  activation_function(const activation_function &orig) = delete;
  virtual ~activation_function() {}

  virtual double compute(const double &val) const = 0;
  virtual double derivative(const double &val) const = 0;
};

class sigmoid : public activation_function
{

public:
  sigmoid() {}
  sigmoid(const sigmoid &orig) = delete;
  ~sigmoid() {}

  double compute(const double &val) const override;
  double derivative(const double &val) const override;
};

class linear : public activation_function
{

public:
  linear() {}
  linear(const linear &orig) = delete;
  ~linear() {}

  double compute(const double &val) const override { return val; }
  double derivative(const double &val) const override { return 1.0; }
};

} // namespace nn