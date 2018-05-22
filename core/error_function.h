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
  training_data(std::vector<double> &input, std::vector<double> &output) : input(input), output(output) {}
  ~training_data() {}
};

class error_function
{
public:
  error_function() {}
  ~error_function() {}

  virtual double error(network &net, std::vector<training_data *> &data) = 0;
  virtual void compute_deltas(network &net, training_data &data) = 0;
};

class mean_squared_error : public error_function
{
public:
  mean_squared_error() {}
  ~mean_squared_error() {}

  double error(network &net, std::vector<training_data *> &data) override;
  void compute_deltas(network &net, training_data &data) override;
};

class cross_entropy : public error_function
{
public:
  cross_entropy() {}
  ~cross_entropy() {}

  double error(network &net, std::vector<training_data *> &data) override;
  void compute_deltas(network &net, training_data &data) override;
};
} // namespace nn