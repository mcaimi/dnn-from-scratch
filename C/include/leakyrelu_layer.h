#include "common.h"
#include "matrix.h"

struct __leakyrelu_t {
  // layer input and output dimensions
  unsigned int input_dimensions;
  unsigned int output_dimensions;

  // input and output values
  double *inputs;
  double *outputs;

  // leaky ReLU parameter
  double alpha;
};
typedef struct __leakyrelu_t leakyrelu;

// create new leakyrelu
leakyrelu *leakyreluCreate(unsigned int, double);
// free leakyrelu
void leakyreluFree(leakyrelu *);
// display info
void leakyreluInfo(leakyrelu *);

// set input values into the leakyrelu layer
void leakyreluFeedIn(leakyrelu *, double *);

// feed-forward matrix multiplication
double *leakyreluFeedForward(leakyrelu *);

// gradient descent
double *leakyreluBackPropagation(leakyrelu *, double *);

