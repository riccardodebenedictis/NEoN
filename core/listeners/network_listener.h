#pragma once

namespace nn
{

class network_listener
{
public:
  network_listener() {}
  network_listener(const network_listener &orig) = delete;
  ~network_listener() {}

  virtual void start_training(const double &error) = 0;
  virtual void stop_training(const double &error) = 0;

  virtual void start_epoch(const double &error) = 0;
  virtual void stop_epoch(const double &error) = 0;
};
} // namespace nn