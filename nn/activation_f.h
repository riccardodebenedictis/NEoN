#pragma once

namespace nn
{

class activation_f
{
public:
  activation_f() {}
  activation_f(const activation_f &orig) = delete;
  virtual ~activation_f() {}

  virtual double compute(const double &val) const = 0;
  virtual double derivative(const double &val) const = 0;
};

class sigmoid : public activation_f
{

public:
  sigmoid() {}
  sigmoid(const sigmoid &orig) = delete;
  ~sigmoid() {}

  double compute(const double &val) const override;
  double derivative(const double &val) const override;
};

class linear : public activation_f
{

public:
  linear() {}
  linear(const linear &orig) = delete;
  ~linear() {}

  double compute(const double &val) const override;
  double derivative(const double &val) const override;
};

} // namespace nn