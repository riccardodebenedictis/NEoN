#include "network.h"
#include <cassert>

using namespace nn;

int main(int argc, char *argv[])
{
    // we create the network..
    mean_squared_error mse;
    sigmoid sgm;
    network net(mse, sgm, {2, 3, 1});

    // we create the training data..
    std::vector<data_row *> tr_data({new data_row({3, 5}, {0.75}),
                                       new data_row({5, 1}, {0.82}),
                                       new data_row({10, 2}, {0.93})});
                                       
    // we create the test data..
    std::vector<data_row *> tst_data({new data_row({3, 5}, {0.75}),
                                       new data_row({5, 1}, {0.82}),
                                       new data_row({10, 2}, {0.93})});

    // this is the current error on training data before the training..
    double c_err = net.get_error(tr_data);

    // we train the network through stochastic gradient descent..
    net.sgd(tr_data, tst_data, 200, 3, 3);

    // this is the current error on training data after the training..
    double t_err = net.get_error(tr_data);
    assert(c_err > t_err);
}