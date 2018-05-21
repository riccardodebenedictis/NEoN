class activation_function
{
  public:
    activation_function();
    ~activation_function();

    double compute(const double &val);
    double derivative();
};