#include "network.h"
#ifndef NDEBUG
#include "socket_listener.h"
#endif
#include <cassert>

using namespace nn;

int main(int argc, char *argv[])
{
    // we create the network..
    cross_entropy ce;
    sigmoid sgm;
    network net(ce, sgm, {2, 4, 4, 1});

    // we create the training data..
    std::vector<training_data *> data({new training_data({0, 0}, {0}),
                                       new training_data({0, 1}, {1}),
                                       new training_data({1, 0}, {1}),
                                       new training_data({1, 1}, {0})});

    // this is the current error on training data before the training..
    double c_err = net.get_error(data);

#ifndef NDEBUG
    socket_listener l;
    net.add_listener(l);
#endif

    // we train the network through stochastic gradient descent..
    net.sgd(data, 20000, 2, 0.005);

    // this is the current error on training data after the training..
    double t_err = net.get_error(data);
    assert(c_err > t_err);
}