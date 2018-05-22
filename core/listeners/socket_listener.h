#pragma once

#include "network_listener.h"
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#endif
#include <string>

namespace nn
{

class socket_listener : public network_listener
{
  private:
#ifdef _WIN32
    SOCKET skt;
#else
    int skt;
#endif

  public:
    socket_listener();
    socket_listener(const socket_listener &orig) = delete;
    ~socket_listener();

    void start_training(const double &error) override;
    void stop_training(const double &error) override;

    void start_epoch(const double &error) override;
    void stop_epoch(const double &error) override;

  private:
    void send_message(const std::string &msg);
};
} // namespace nn