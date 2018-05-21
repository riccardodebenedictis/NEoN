#include "network.h"

int main(int argc, char *argv[])
{
    nn::sigmoid sgm;
    nn::network n(sgm, {2, 3, 1});
    std::vector<double> o = n.forward({1, 1});
}