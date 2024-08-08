#include <math.h>
#include "sigmoid_layer.h"
#include "common.h"

void sigmoidFeedIn(sigmoid *n, double *input) {
  // copy input values into the sigmoid's own input buffer
  copyVectors(input, n->inputs, n->input_dimensions);
}

double *sigmoidFeedForward(sigmoid *n) {
  // calculate sigmoid
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    // sigmoid => S(x) = 1/(1+e^(-x)), compress input between 0 and 1
    n->outputs[o] = 1.0f / (1.0f + exp(-(n->inputs[o])));
  }

  // return values
  return n->outputs;
}
