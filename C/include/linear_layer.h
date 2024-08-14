#include "common.h"
#include "matrix.h"

struct __linear_t {
  // layer input and output dimensions
  unsigned int input_dimensions;
  unsigned int output_dimensions;

  // input and output values
  double *inputs;
  double *outputs;

  // i*o matrix
  double **weights_matrix;
  double *bias;

  // learning rate
  double learning_rate;
};
typedef struct __linear_t linear;

// create new linear
linear *linearCreate(unsigned int, unsigned int, double);
// free linear
void linearFree(linear *);
// display info
void linearInfo(linear *);

// set learning rate
void linearSetLR(linear *, double);
// save linear checkpoint
void linearSaveCheckpoint(linear *, char *);
// load linear checkpoint
linear *linearLoadCheckpoint(char *);

// set input values into the linear layer
void linearFeedIn(linear *, double *);

// feed-forward matrix multiplication
double *linearFeedForward(linear *);

// gradient descent
double *linearBackPropagation(linear *, double *);

