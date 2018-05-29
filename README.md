# NEoN

NEoN is a minimalistic artificial NEural Network framework written in C++ which has been designed to be extremely simple to use.

Unlike other approaches, NEoN allows the training of the same network starting from different training data so as to allow its use in a Reinforcement Learning (RL) context.

## Getting started

The basic steps for using NEoN are:
1. Create a neural network
2. Create the training (and evaluation) dataset
3. Train the network
4. Use the network

## Building NEoN

The basic requirements for building NEoN are:

- [Git](https://git-scm.com/)
- [CMake](https://cmake.org) v3.x
- A C++ compiler

### Building on Linux

The easiest way to install the building requirements on Ubuntu is as follows

```
sudo apt-get install build-essential
sudo apt-get install cmake
```

once the building requirements are installed, move to a desired folder and clone the NEoN repository

```
git clone https://github.com/riccardodebenedictis/NEoN
```

finally, build NEoN

```
mkdir build
cd build
cmake ..
make
```