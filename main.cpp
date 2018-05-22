#include "network.h"

int main(int argc, char *argv[])
{
    nn::mean_squared_error mse;
    nn::sigmoid sgm;
    nn::network n(mse, sgm, {2, 3, 1});
    std::vector<double> o = n.forward({1, 1});
}