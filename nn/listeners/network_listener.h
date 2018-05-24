#pragma once

#include <cstddef>

namespace nn
{

class network_listener
{
public:
  network_listener() {}
  network_listener(const network_listener &orig) = delete;
  ~network_listener() {}

  virtual void start_training(const std::size_t &n_epochs, const double &tr_error, const double &tst_error) = 0;
  virtual void stop_training(const double &tr_error, const double &tst_error) = 0;

  virtual void start_epoch(const double &tr_error, const double &tst_error) = 0;
  virtual void stop_epoch(const double &tr_error, const double &tst_error) = 0;
};
} // namespace nn