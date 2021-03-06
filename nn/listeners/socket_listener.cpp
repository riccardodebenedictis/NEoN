#include "socket_listener.h"
#include <iostream>

namespace nn
{
socket_listener::socket_listener()
{
#ifdef _WIN32
    WSADATA wsa_data;
    int err_c = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (err_c != 0)
        std::cerr << "WSAStartup failed with error: " << std::to_string(err_c) << std::endl;
#endif

    skt = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

#ifdef _WIN32
    if (skt == INVALID_SOCKET)
#else
    if (skt < 0)
#endif
        std::cerr << "unable to connect to server.." << std::endl;

    struct sockaddr_in sa;
    sa.sin_family = AF_INET;
    sa.sin_port = htons(1100);
    inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);

    if (connect(skt, (struct sockaddr *)&sa, sizeof(sa)) < 0)
        std::cerr << "unable to connect to server.." << std::endl;
}

socket_listener::~socket_listener()
{
#ifdef _WIN32
    closesocket(skt);
    int err_c = WSACleanup();
#else
    close(skt);
#endif
}

void socket_listener::start_training(const std::size_t &n_epochs, const double &tr_error, const double &tst_error) { send_message("strt_tr{\"n_epcs\":" + std::to_string(n_epochs) + ", \"tr_error\":" + std::to_string(tr_error) + ", \"tst_error\":" + std::to_string(tst_error) + "}\n"); }
void socket_listener::stop_training(const double &tr_error, const double &tst_error) { send_message("stp_tr{\"tr_error\":" + std::to_string(tr_error) + ", \"tst_error\":" + std::to_string(tst_error) + "}\n"); }

void socket_listener::start_epoch(const double &tr_error, const double &tst_error) { send_message("strt_epc{\"tr_error\":" + std::to_string(tr_error) + ", \"tst_error\":" + std::to_string(tst_error) + "}\n"); }
void socket_listener::stop_epoch(const double &tr_error, const double &tst_error) { send_message("stp_epc{\"tr_error\":" + std::to_string(tr_error) + ", \"tst_error\":" + std::to_string(tst_error) + "}\n"); }

void socket_listener::send_message(const std::string &msg)
{
    int total = 0;
    int len = static_cast<int>(msg.size());
    int bytesleft = len;
    int n = -1;
    while (total < len)
    {
        n = send(skt, msg.c_str() + total, bytesleft, 0);
        if (n <= 0)
            throw(n);
        total += n;
        bytesleft -= n;
    }
}
} // namespace nn