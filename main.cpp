#include "layer.h"

int main(int argc, char *argv[])
{
    nn::layer l(4, 3);
    nn::matrix input(3, 2);
    nn::matrix output = l.forward(input);
}