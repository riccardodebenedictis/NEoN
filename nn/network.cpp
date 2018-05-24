#include "network.h"

namespace nn
{

layer::layer(std::default_random_engine &gen, const std::size_t &lr_size, const std::size_t &nr_size) : lr_size(lr_size), nr_size(nr_size), w(lr_size, std::vector<double>(nr_size)), b(lr_size), z(lr_size), a(lr_size), nabla_w(lr_size, std::vector<double>(nr_size)), nabla_b(lr_size)
{
    std::normal_distribution<double> n_dist(0, 1);
    for (std::size_t i = 0; i < lr_size; ++i)
    {
        for (std::size_t j = 0; j < nr_size; ++j)
            w[i][j] = n_dist(gen) / std::sqrt(nr_size);
        b[i] = n_dist(gen);
    }
}

layer::~layer() {}

std::vector<double> layer::forward(const activation_f &af, const std::vector<double> &x)
{
    for (std::size_t i = 0; i < lr_size; ++i)
    {
        z[i] = b[i];
        for (std::size_t j = 0; j < nr_size; ++i)
            z[i] += w[i][j] * x[i];
        a[i] = af.compute(z[i]);
    }
    return a;
}

network::network(error_f &ef, activation_f &af, const std::vector<std::size_t> &sizes) : ef(ef), af(af), layers(sizes.size() - 1), delta(sizes.size() - 1), size(sizes.size() - 1)
{
    for (std::size_t i = 0; i < sizes.size() - 1; ++i)
        layers[i] = new layer(gen, sizes[i + 1], sizes[i]);
}

network::~network()
{
    for (std::size_t i = 0; i < layers.size(); ++i)
        delete layers[i];
}

std::vector<double> network::forward(const std::vector<double> &x)
{
    std::vector<double> a = layers[0]->forward(af, x);
    for (std::size_t i = 1; i < layers.size(); ++i)
        a = layers[i]->forward(af, a);
    return a;
}

void network::sgd(std::vector<data_row *> &tr_data, std::vector<data_row *> &eval_data, const std::size_t &epochs, const std::size_t &mini_batch_size, const double &eta, const double &lambda)
{
    for (std::size_t i = 1; i <= epochs; ++i)
    {
        // we shuffle the training data..
        std::shuffle(tr_data.begin(), tr_data.end(), gen);
        // we partition the training data into mini batches of 'mini_batch_size' size..
        for (std::size_t j = 0; j <= tr_data.size() - mini_batch_size; j += mini_batch_size)
            update_mini_batch(std::vector<data_row *>(tr_data.begin() + j, tr_data.begin() + j + mini_batch_size), eta, lambda);
    }
}

void network::update_mini_batch(const std::vector<data_row *> &mini_batch, const double &eta, const double &lambda)
{
    // we perform backpropagation..
    for (data_row *data : mini_batch)
        backprop(*data);

    // we update the biases, the weigths, and clean up things..
    for (std::size_t i = 0; i < size; ++i)
        for (std::size_t j = 0; j < layers[i]->lr_size; ++j)
        {
            layers[i]->b[j] -= (eta / mini_batch.size()) * layers[i]->nabla_b[j];
            layers[i]->nabla_b[j] = 0;
            for (std::size_t k = 0; k < layers[i]->nr_size; ++k)
                layers[i]->w[j][k] -= (eta / mini_batch.size()) * layers[i]->nabla_w[j][k] + ((eta * lambda) / mini_batch.size()) * layers[i]->w[j][k];
            layers[i]->nabla_w[j].assign(layers[i]->nr_size, 0);
        }
}

void network::backprop(const data_row &data)
{
    // feedforward..
    std::vector<double> a = forward(data.x);

    // we compute the deltas for the output layer..
    delta[layers.size() - 1] = ef.delta(af, layers[layers.size() - 1]->z, a, data.y);

    // we compute the deltas for the other layers..
    for (std::size_t i = size - 1; i > 0; --i)
        for (std::size_t j = 0; j < layers[i - 1]->lr_size; ++j)
        {
            delta[i - 1][j] = 0;
            for (std::size_t k = 0; k < layers[i]->lr_size; ++k)
                delta[i - 1][j] += layers[i]->w[k][j] * delta[i][k];
        }

    // we use the computed deltas to update the nablas..
    for (std::size_t i = size - 1; i >= 1; --i)
        for (std::size_t j = 0; j < layers[i]->lr_size; ++j)
        {
            layers[i]->nabla_b[j] += delta[i][j];
            for (std::size_t k = 0; k < layers[i - 1]->lr_size; ++k)
                layers[i]->nabla_w[j][k] += layers[i - 1]->a[k] * delta[i][j];
        }

    for (std::size_t i = 0; i < layers[0]->lr_size; ++i)
    {
        layers[0]->nabla_b[i] += delta[0][i];
        for (std::size_t k = 0; k < data.x.size(); ++k)
            layers[0]->nabla_w[i][k] += data.x[k] * delta[0][i];
    }
}
} // namespace nn