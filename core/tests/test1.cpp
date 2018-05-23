#include "network.h"
#include <cassert>

using namespace nn;

int main(int argc, char *argv[])
{
    // we create the network..
    cross_entropy ce;
    sigmoid sgm;
    network net(ce, sgm, {2, 4, 4, 1});

    // we create the training data..
    std::vector<data_row *> tr_data({new data_row({0, 0}, {0}),
                                       new data_row({0, 1}, {1}),
                                       new data_row({1, 0}, {1}),
                                       new data_row({1, 1}, {0})});

    // we create the test data..
    std::vector<data_row *> tst_data({new data_row({0, 0}, {0}),
                                       new data_row({0, 1}, {1}),
                                       new data_row({1, 0}, {1}),
                                       new data_row({1, 1}, {0})});

    // this is the current error on training data before the training..
    double c_err = net.get_error(tr_data);

    // we train the network through stochastic gradient descent..
    net.sgd(tr_data, tst_data, 2000, 2, 0.05);

    // this is the current error on training data after the training..
    double t_err = net.get_error(tr_data);
    assert(c_err > t_err);
}