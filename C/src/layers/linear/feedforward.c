#include "linear_layer.h"
#include "common.h"
#include "matmult.h"

void linearFeedIn(linear *n, double *input) {
  // copy input values into the linear's own input buffer
  copyVectors(input, n->inputs, n->input_dimensions);
}

double *linearFeedForward(linear *n) {
  // multiply weights matrix and input vector
  double *outvec = vec2Mat(n->inputs, n->weights_matrix, n->input_dimensions, n->output_dimensions);

  // add biases
  for (unsigned int b=0; b<n->output_dimensions; b++) {
    outvec[b] += n->bias[b];
  }

  // return values
  copyVectors(outvec, n->outputs, n->output_dimensions);
  free(outvec);
  return n->outputs;
}
