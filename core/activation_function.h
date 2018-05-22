#pragma once

namespace nn
{

class activation_function
{
public:
  activation_function() {}
  virtual ~activation_function() {}

  virtual double compute(const double &val) = 0;
  virtual double derivative(const double &val) = 0;
};

class sigmoid : public activation_function
{

public:
  sigmoid() {}
  ~sigmoid() {}

  double compute(const double &val) override;
  double derivative(const double &val) override;
};

class linear : public activation_function
{

public:
  linear() {}
  ~linear() {}

  double compute(const double &val) override { return val; }
  double derivative(const double &val) override { return 1.0; }
};

} // namespace nn