#include "common.h"

struct __relu_t {
  // layer input and output dimensions
  unsigned int input_dimensions;
  unsigned int output_dimensions;

  // input and output values
  double *inputs;
  double *outputs;
};
typedef struct __relu_t relu;

// create new relu
relu *reluCreate(unsigned int);
// free relu
void reluFree(relu *);
// display info
void reluInfo(relu *);

// set input values into the relu layer
void reluFeedIn(relu *, double *);

// feed-forward matrix multiplication
double *reluFeedForward(relu *);

// gradient descent
double *reluBackPropagation(relu *, double *);

