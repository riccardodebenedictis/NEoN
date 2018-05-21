#include <cstddef>
#include <vector>
#include <cassert>

namespace nn
{

class matrix
{
private:
  const std::size_t r;                 // the number of rows..
  const std::size_t c;                 // the number of columns..
  std::vector<std::vector<double>> _m; // the matrix..

public:
  matrix(const std::size_t &r, const std::size_t &c) : r(r), c(c), _m(r, std::vector<double>(c)) {}
  ~matrix() {}

  const std::size_t &rows() const { return r; };    // the number of rows..
  const std::size_t &columns() const { return c; }; // the number of columns..

  double get(const std::size_t &r, const std::size_t &c) const { return _m[r][c]; }
  void set(const std::size_t &r, const std::size_t &c, const double &v) { _m[r][c] = v; }

  matrix operator+(const matrix &rhs) const
  {
    if (r != rhs.r || c != rhs.c)
    {
    }
    else
    {
      matrix sum(r, c);
      for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
          sum._m[i][j] += _m[i][j] + rhs._m[i][j];
      return sum;
    }
  }

  matrix operator*(const matrix &rhs) const
  {
    matrix prod(rhs.r, c);

    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < rhs.c; ++j)
        for (size_t k = 0; k < r; ++k)
          prod._m[i][j] += _m[i][k] * rhs._m[k][j];

    return prod;
  }
};

} // namespace nn