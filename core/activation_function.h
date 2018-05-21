namespace nn
{

class activation_function
{
public:
  activation_function() {}
  virtual ~activation_function() {}

  virtual double compute(const double &val) = 0;
  virtual double derivative() = 0;
};

class sigmoid : public activation_function
{
private:
  double output;

public:
  sigmoid() {}
  ~sigmoid() {}

  double compute(const double &val) override;
  double derivative() override;
};

class linear : public activation_function
{
private:
  double output;

public:
  linear() {}
  ~linear() {}

  double compute(const double &val) override { return val; }
  double derivative() override { return 1.0; }
};

} // namespace nn