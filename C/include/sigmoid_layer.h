#include "common.h"
#include "matrix.h"

struct __sigmoid_t {
  // layer input and output dimensions
  unsigned int input_dimensions;
  unsigned int output_dimensions;

  // input and output values
  double *inputs;
  double *outputs;
};
typedef struct __sigmoid_t sigmoid;

// create new sigmoid
sigmoid *sigmoidCreate(unsigned int);
// free sigmoid
void sigmoidFree(sigmoid *);
// display info
void sigmoidInfo(sigmoid *);

// set input values into the sigmoid layer
void sigmoidFeedIn(sigmoid *, double *);

// feed-forward matrix multiplication
double *sigmoidFeedForward(sigmoid *);

// gradient descent
double *sigmoidBackPropagation(sigmoid *, double *);

