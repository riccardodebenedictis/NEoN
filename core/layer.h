#include "matrix.h"

namespace nn
{

class layer
{
private:
  matrix _w;
  matrix _b;

public:
  layer(const std::size_t &lr_size, const std::size_t &nr_size);
  ~layer();

  matrix forward(const matrix &input);
};

} // namespace nn